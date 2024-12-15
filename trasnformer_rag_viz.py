# Filename: transformer_rag.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from utils import load_data_from_csv, preprocess_data
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import pytorch_cos_sim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

## KNOWLEDGE BASE CONSTRUCTION

# Loading from the 3.7K code csv file
knowledge_base_df = pd.read_csv("3.7K_python_dataset.csv")

# Extract intents and snippets
intents = knowledge_base_df['Problem'].tolist()
snippets = knowledge_base_df['Python Code'].tolist()

# Initialize a dense retrieval model
retrieval_model = SentenceTransformer('all-mpnet-base-v2')

# Embed knowledge base
kb_embeddings = retrieval_model.encode(intents, convert_to_tensor=True)

## RETRIEVAL FUNCTION
def retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=3):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb_embeddings)
    top_k_indices = torch.topk(similarities, k=top_k).indices[0].tolist()

    # Retrieve top-k documents
    retrieved_intents = [intents[i] for i in top_k_indices]
    retrieved_snippets = [snippets[i] for i in top_k_indices]

    retrieved_contexts = [
        f"Problem: {intent} | Code: {snippet}"
        for intent, snippet in zip(retrieved_intents, retrieved_snippets)
    ]

    return retrieved_contexts

## AUGMENT OUTPUT
def augment_query(query, retrieved_docs, max_len=256):
    context = " ".join(retrieved_docs)
    truncated_context = context[:max_len - len(query) - len("Context: Query: ")]
    return f"Context: {context} Query: {query}"

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Load and preprocess data
data = load_data_from_csv('mbpp_conala.csv')
train_df, val_df, test_df = preprocess_data(data)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dataset Class
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.df.iloc[index]['intent']
        target_text = self.df.iloc[index]['snippet']

        retrieved_docs = retrieve_knowledge(source_text, kb_embeddings, intents, snippets)
        augmented_source = augment_query(source_text, retrieved_docs, max_len=self.max_len)

        # Construct the final prompt
        ag = "Generate to python using the following context" + augmented_source

        source_encoding = self.tokenizer(
            ag,
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

        labels = target_encoding['input_ids'].squeeze(dim=0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(dim=0),
            'attention_mask': source_encoding['attention_mask'].squeeze(dim=0),
            'labels': labels,
        }

train_dataset = CodeDataset(train_df, tokenizer)
val_dataset = CodeDataset(val_df, tokenizer)
test_dataset = CodeDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=20e-5)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", disable=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(data_loader)

# Training loop with loss tracking
epochs = 4
train_losses = []
val_losses = []

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = eval_epoch(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')

def generate_code_from_random_queries(model, tokenizer, kb_embeddings, intents, snippets, test_df, device, top_k=3):
    query = "Python Program to Find the Fibonacci Series without Using Recursion"
    random_queries = random.sample(test_df['intent'].tolist(), 2)
    random_queries.append(query)

    for query in random_queries:
        print(f"Query: {query}")
        retrieved_docs = retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=top_k)
        augmented_query = augment_query(query, retrieved_docs)
        
        encoding = tokenizer(augmented_query, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            output = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=256)

        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"Retrieved docs:\n")
        for doc in retrieved_docs:
            print(f"- {doc}")
        print(f"Generated Python Code: {decoded_output}")
        print("-" * 80)

generate_code_from_random_queries(model, tokenizer, kb_embeddings, intents, snippets, test_df, device)

# Save the fine-tuned model
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')

# Plot training and validation loss curves
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss', marker='o')
plt.title('Train/Validation Loss over Epochs (RAG Transformer)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('transformer_rag_loss_curves.png')
plt.show()