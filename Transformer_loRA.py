# Filename: transformer_model_lora.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from utils import load_data, load_data_from_csv, preprocess_data, preprocess_data_new
import logging
import random
from tqdm import tqdm

# LoRA imports
from peft import LoraConfig, get_peft_model

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Load and preprocess data
data = load_data_from_csv('3.7K_python_dataset.csv')
train_df, val_df, test_df = preprocess_data_new(data)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

########################################
# Apply LoRA
########################################
lora_config = LoraConfig(
    r=16,           # LoRA rank
    lora_alpha=32,  # Alpha scaling factor
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Optional: to verify LoRA params

class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_text = self.df.iloc[index]['Problem']
        target_text = self.df.iloc[index]['Python Code']

        source = "Translate to Python: " + source_text

        source_encoding = self.tokenizer(
            source,
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

        labels = target_encoding['input_ids'].squeeze().long()
        labels[labels == self.tokenizer.pad_token_id] = -100 

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels,
        }

train_dataset = CodeDataset(train_df, tokenizer)
val_dataset = CodeDataset(val_df, tokenizer)
test_dataset = CodeDataset(test_df, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)

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

def eval_epoch(model, data_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
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

            generated_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)
            batch_predictions = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            labels[labels == -100] = tokenizer.pad_token_id
            batch_references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(batch_predictions)
            references.extend(batch_references)

    for i in range(min(3, len(predictions))):
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 50)       

    return total_loss / len(data_loader), predictions, references

epochs = 5
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, predictions, references = eval_epoch(model, val_loader, tokenizer, device)

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')

print("\nTESTING WITH QUERY\n")

def test_random_queries(model, tokenizer, test_df, device, top_k=3):
    query = "Generate the first n fibonacci numbers"
    random_queries = random.sample(test_df['Problem'].tolist(), 2)
    random_queries.append(query)
    for query in random_queries:
        print(f"Query: {query}")
        encoding = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output = model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'], max_length=256)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Python Code: {decoded_output}")
        print("-" * 80)

test_random_queries(model, tokenizer, test_dataset.df, device)

# Save the LoRA fine-tuned model (LoRA adapters + base model)
model.save_pretrained('./finetuned_lora_model')
tokenizer.save_pretrained('./finetuned_lora_model')