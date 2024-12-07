import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import logging
from tqdm import tqdm
from utils import load_data, preprocess_data

# **SET UP LOGGING**
logging.basicConfig(level=logging.INFO)

# **KNOWLEDGE BASE CONSTRUCTION**
# Load knowledge base from the 600k dataset
knowledge_base_df = pd.read_csv("600kdataset_table.csv")
intents = knowledge_base_df['intent'].tolist()
snippets = knowledge_base_df['snippet'].tolist()

# Initialize dense retrieval model
retrieval_model = SentenceTransformer('all-mpnet-base-v2').to(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Embed knowledge base
logging.info("Creating embeddings for the knowledge base...")
kb_embeddings = []
batch_size = 64
for i in tqdm(range(0, len(intents), batch_size), desc="Embedding KB"):
    batch_intents = intents[i:i + batch_size]
    batch_embeddings = retrieval_model.encode(batch_intents, convert_to_tensor=True).cpu().numpy()
    kb_embeddings.append(batch_embeddings)
kb_embeddings = np.vstack(kb_embeddings)

# **BUILD FAISS INDEX**
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
assert index.ntotal == len(kb_embeddings), "Mismatch between FAISS index and embeddings count!"

# Save the FAISS index
faiss.write_index(index, "knowledge_base.index")

# Reload the FAISS index
index = faiss.read_index("knowledge_base.index")

# **RETRIEVAL FUNCTION**
def retrieve_knowledge_faiss(query, faiss_index, intents, snippets, top_k=3):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True).cpu().numpy()
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), top_k)
    
    retrieved_intents = [intents[i] for i in indices[0]]
    retrieved_snippets = [snippets[i] for i in indices[0]]
    
    retrieved_contexts = [
        f"Problem: {intent} | Code: {snippet}"
        for intent, snippet in zip(retrieved_intents, retrieved_snippets)
    ]
    return retrieved_contexts

# **AUGMENT QUERY**
def augment_query(query, retrieved_docs, max_len=256):
    context = " ".join(retrieved_docs)
    truncated_context = context[:max_len - len(query) - len("Context: Query: ")]
    return f"Context: {truncated_context} Query: {query}"

# **DATASET AND DATALOADER**
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, faiss_index, intents, snippets, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.index = faiss_index
        self.intents = intents
        self.snippets = snippets
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        source_text = self.df.iloc[idx]['intent']
        target_text = self.df.iloc[idx]['snippet']
        
        retrieved_docs = retrieve_knowledge_faiss(source_text, self.index, self.intents, self.snippets)
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
        
        labels = target_encoding['input_ids'].squeeze(dim=0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(dim=0),
            'attention_mask': source_encoding['attention_mask'].squeeze(dim=0),
            'labels': labels,
        }

# **PREPROCESS DATA**
data = load_data()
train_df, val_df, test_df = preprocess_data(data)

# **INITIALIZE TOKENIZER AND MODEL**
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base').to(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# **CREATE DATASETS AND DATALOADERS**
train_dataset = CodeDataset(train_df, tokenizer, index, intents, snippets)
val_dataset = CodeDataset(val_df, tokenizer, index, intents, snippets)
test_dataset = CodeDataset(test_df, tokenizer, index, intents, snippets)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# **TRAINING LOOP**
optimizer = AdamW(model.parameters(), lr=5e-5)

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    return total_loss / len(data_loader)

# **TRAINING PROCESS**
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = eval_epoch(model, val_loader, device)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

# **SAVE THE FINE-TUNED MODEL**
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')
