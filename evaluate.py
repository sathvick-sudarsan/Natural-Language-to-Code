# Filename: evaluate.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import logging

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

########################################
# Utility Functions
########################################

def load_data_from_csv(csv_file_path):
    data = pd.read_csv(csv_file_path)
    return data

def preprocess_data(data):
    from sklearn.model_selection import train_test_split
    data.drop_duplicates(subset=['intent', 'snippet'], inplace=True)
    data.dropna(subset=['intent', 'snippet'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = str(self.df.iloc[index]['intent'])
        target_text = str(self.df.iloc[index]['snippet'])

        source = "Translate to Python: " + source_text

        source_encoding = self.tokenizer(
            source,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # These are shape: [1, seq_len]
        # Squeeze them so they become [seq_len]. The DataLoader with batch_size=1 will add back the batch dimension.
        input_ids = source_encoding['input_ids'].squeeze(0)          # shape: [seq_len]
        attention_mask = source_encoding['attention_mask'].squeeze(0)# shape: [seq_len]

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

    test_dataset = CodeDataset(test_df, tokenizer, max_len=256)
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
        # Now batch['input_ids'] is shape [batch_size, seq_len] = [1, seq_len] already
        input_ids = batch['input_ids'].to(device)         # shape: [1, seq_len]
        attention_mask = batch['attention_mask'].to(device)# shape: [1, seq_len]
        target_text = batch['target_text'][0]

        # Generate code
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
