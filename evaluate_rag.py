# Filename: evaluate_rag.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Import utilities from your utils.py
from utils import load_data_from_csv, preprocess_data

########################################
# Retrieval and Augmentation Functions
########################################

# Load the larger knowledge base
knowledge_base_df = pd.read_csv("3.7K_python_dataset.csv")
intents = knowledge_base_df['Problem'].tolist()
snippets = knowledge_base_df['Python Code'].tolist()

retrieval_model = SentenceTransformer('all-mpnet-base-v2')
kb_embeddings = retrieval_model.encode(intents, convert_to_tensor=True)

def retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=3):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb_embeddings)
    top_k_indices = torch.topk(similarities, k=top_k).indices[0].tolist()

    retrieved_intents = [intents[i] for i in top_k_indices]
    retrieved_snippets = [snippets[i] for i in top_k_indices]

    retrieved_contexts = [
        f"Problem: {intent} | Code: {snippet}"
        for intent, snippet in zip(retrieved_intents, retrieved_snippets)
    ]
    return retrieved_contexts

def augment_query(query, retrieved_docs, max_len=256):
    context = " ".join(retrieved_docs)
    return f"Context: {context} Query: {query}"

########################################
# Dataset Class
########################################

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, kb_embeddings, intents, snippets, retrieval_model, max_len=256, top_k=3):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.kb_embeddings = kb_embeddings
        self.intents = intents
        self.snippets = snippets
        self.retrieval_model = retrieval_model
        self.top_k = top_k

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.df.iloc[index]['intent']
        target_text = self.df.iloc[index]['snippet']

        retrieved_docs = retrieve_knowledge(source_text, self.kb_embeddings, self.intents, self.snippets, top_k=self.top_k)
        augmented_source = augment_query(source_text, retrieved_docs, max_len=self.max_len)

        source_encoding = self.tokenizer(
            augmented_source,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Squeeze to remove the batch dimension here, so we get [seq_len]
        input_ids = source_encoding['input_ids'].squeeze(0)
        attention_mask = source_encoding['attention_mask'].squeeze(0)

        labels = target_encoding['input_ids'].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_text': target_text
        }

########################################
# Main Evaluation Code
########################################

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data_from_csv('mbpp_conala.csv')
    _, _, test_df = preprocess_data(data)

    # Load model and tokenizer
    model_dir = './finetuned_model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_dataset = CodeDataset(test_df, tokenizer, kb_embeddings, intents, snippets, retrieval_model, max_len=256, top_k=3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    references = []

    # For BLEU
    smoothie = SmoothingFunction().method4
    bleu_scores = []

    # For ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []

    # Ensure pad token is defined
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    print("Evaluating on test set...")

    for batch in test_loader:
        # batch_size=1, so batch['input_ids'] and batch['attention_mask'] are [1, seq_len]
        input_ids = batch['input_ids'].to(device)          # shape: [1, seq_len]
        attention_mask = batch['attention_mask'].to(device)# shape: [1, seq_len]
        target_text = batch['target_text'][0]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(pred_text)
        references.append(target_text)

        # Compute BLEU
        ref_tokens = nltk.word_tokenize(target_text)
        pred_tokens = nltk.word_tokenize(pred_text)
        bleu_score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

        # Compute ROUGE
        rouge_scores = scorer.score(target_text, pred_text)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougel_scores.append(rouge_scores['rougeL'].fmeasure)

    # Compute average BLEU
    avg_bleu = np.mean(bleu_scores)

    # Compute average ROUGE
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougel = np.mean(rougel_scores)

    # Exact match accuracy
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    exact_match_accuracy = exact_matches / len(predictions)

    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
    print(f"ROUGE-L F1: {avg_rougel:.4f}")
    print(f"Exact Match Accuracy: {exact_match_accuracy*100:.2f}%")

    # Print sample predictions
    print("\nSample Predictions:")
    for i in range(min(3, len(predictions))):
        print(f"Intent: {test_df.iloc[i]['intent']}")
        print(f"Reference Code:\n{references[i]}")
        print(f"Predicted Code:\n{predictions[i]}\n")
