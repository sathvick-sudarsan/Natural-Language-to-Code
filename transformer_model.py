# Filename: transformer_model.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from utils import load_data, preprocess_data
import logging

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

        #labels = target_encoding['input_ids'].squeeze()
        #labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss computation
        labels = target_encoding['input_ids'].squeeze().long()  # Ensure labels are of type torch.long
        labels[labels == self.tokenizer.pad_token_id] = -100 

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
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

def eval_epoch(model, data_loader, tokenizer,device):
    model.eval()
    total_loss = 0
    predictions = [] ## added to visualise output
    references = []  ## added to visualise output 

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

            ## ADDED THIS PART TO GENERATE PREDICTIONS  
            generated_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)
            batch_predictions = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            labels[labels == -100] = tokenizer.pad_token_id  # Replace -100 with pad token ID
            batch_references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(batch_predictions)
            references.extend(batch_references)
    ## PRINTING THE OUTPUTS     
    for i in range(min(3, len(predictions))):
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 50)       

    return total_loss / len(data_loader),predictions,references

# Training loop
epochs = 5
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, predictions, references = eval_epoch(model, val_loader,tokenizer, device)

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')

   ## Save predictions and references for analysis (optional)[MOSTLY NO NEED]
    #with open(f'epoch_{epoch+1}_predictions.txt', 'w') as pred_file, \
         #open(f'epoch_{epoch+1}_references.txt', 'w') as ref_file:
        #for p, r in zip(predictions, references):
            #pred_file.write(p + '\n')
            #ref_file.write(r + '\n')

# Save the fine-tuned model
model.save_pretrained('./finetuned_model')
tokenizer.save_pretrained('./finetuned_model')


