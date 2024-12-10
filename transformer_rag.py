# Filename: transformer_model.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from utils import load_data, preprocess_data
import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import pytorch_cos_sim
import random 


## KNOWLEDGE BASE CONSTRUCTIO

# Loading from the 600k code csv files.
knowledge_base_df = pd.read_csv("3.7K_python_dataset.csv")

# Extracting the intents and snippets ## CHANGED INTENT AND SNIPPET TO THE CSV FILE 
intents = knowledge_base_df['Problem'].tolist()  # Problems in Python
snippets = knowledge_base_df['Python Code'].tolist()

# Initialize a dense retrieval model
retrieval_model = SentenceTransformer('all-mpnet-base-v2')

# Embed knowledge base
kb_embeddings = retrieval_model.encode(intents, convert_to_tensor=True)


## RETRIEVAL FUNCTION 

from sentence_transformers.util import pytorch_cos_sim

def retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=3):
    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, kb_embeddings) # Replaced pytorch_cos_sim
    top_k_indices = torch.topk(similarities, k=top_k).indices[0].tolist()

    # Retrieve intents and corresponding snippets
    retrieved_intents = [intents[i] for i in top_k_indices]
    retrieved_snippets = [snippets[i] for i in top_k_indices]

    # Combine intents and snippets for context
    retrieved_contexts = [
        f"Problem: {intent} | Code: {snippet}"
        for intent, snippet in zip(retrieved_intents, retrieved_snippets)
    ]

    return retrieved_contexts



## AUGMENT OUTPUT

def augment_query(query, retrieved_docs, max_len = 256):
    context = " ".join(retrieved_docs)
    truncated_context = context[:max_len - len(query) - len("Context: Query: ")] ## USE THIS IF THERE IS AN ISSUE WITH SIZE 
    return f"Context: {context} Query: {query}"


# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Load and preprocess data
data = load_data()
train_df, val_df, test_df = preprocess_data(data)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Dataset and DataLoader remain the same
class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.df.iloc[index]['intent']
        target_text = self.df.iloc[index]['snippet']

        # Retrieve additional context
        retrieved_docs = retrieve_knowledge(source_text, kb_embeddings,intents,snippets)
        # Check if you need to use source source_text instead of source
        augmented_source = augment_query(source_text, retrieved_docs, max_len = self.max_len)

        source = "Translate to Python: " + source_text 

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
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss computation

        return {
            'input_ids': source_encoding['input_ids'].squeeze(dim=0),
            'attention_mask': source_encoding['attention_mask'].squeeze(dim=0),
            'labels': labels,
        }

# Create datasets and dataloaders
train_dataset = CodeDataset(train_df, tokenizer)
val_dataset = CodeDataset(val_df, tokenizer)
test_dataset = CodeDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training loop
from tqdm import tqdm

optimizer = AdamW(model.parameters(), lr=20e-5)

def train_epoch(model, data_loader, optimizer, device, log_retrieved_docs=True):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        ## Uncomment if output needs to be printed at every stage.
        #if log_retrieved_docs:
        #    source_texts = data_loader.dataset.df.iloc[batch_idx * data_loader.batch_size: (batch_idx + 1) * data_loader.batch_size
        #    ]['intent'].tolist()
        
        #for idx, source_text in enumerate(source_texts):
        #    retrieved_docs = retrieve_knowledge(source_text, kb_embeddings, intents, snippets)
        #    print(f"Batch {batch_idx}, Sample {idx}:\nQuery: {source_text}\nRetrieved Docs: {retrieved_docs}\n")


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
        for batch in tqdm(data_loader, desc="Evaluating",disable = True):
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

# Training loop
epochs = 4
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = eval_epoch(model, val_loader, device)

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')


def generate_code_from_random_queries(model, tokenizer, kb_embeddings, intents, snippets, test_df, device, top_k=3):
    # Select two random queries from the test dataset (or any other source)
    query = "Python Program to Find the Fibonacci Series without Using Recursion"
    random_queries = random.sample(test_df['intent'].tolist(), 2)  # Assuming 'intent' column contains the queries
    
    # Add your predefined query as the last item
    random_queries.append(query)
    
    for query in random_queries:
        print(f"Query: {query}")
        
        # Step 1: Retrieve relevant code snippets for the query
        retrieved_docs = retrieve_knowledge(query, kb_embeddings, intents, snippets, top_k=top_k)
        
        # Step 2: Augment the query with the retrieved context
        augmented_query = augment_query(query, retrieved_docs)
        
        # Step 3: Tokenize the augmented query
        encoding = tokenizer(augmented_query, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        
        # Step 4: Pass the tokenized input through the fine-tuned model to generate code
        with torch.no_grad():
            output = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=256)
        
        # Step 5: Decode the output into readable Python code
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Print the generated code
        print(f"Retrieved docs: \n")
        for doc in retrieved_docs:
          print(f"- {doc}")
        print(f"Generated Python Code: {decoded_output}")
        print("-" * 80)

# Example usage: generate code from two random queries in the test set + one custom query
generate_code_from_random_queries(model, tokenizer, kb_embeddings, intents, snippets, test_df, device)




# Save the fine-tuned model
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')

## changed the disable function for tqdm by adding disable = True 
## Commit 4 : Changed the RAG dataset to MBPP (Gave higher losses compared to 600k)
## Commit 5 : Changed RAG dataset to new python dataset 
## Commit 6 : The other RAG dataset had some issue because of some formatting issue,
            ## So using the downloaded data set with diff row names