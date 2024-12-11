# Filename: transformer_model_lora_optuna.py

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from utils import load_data_from_csv, preprocess_data_new
import logging
import random
from tqdm import tqdm

# LoRA imports
from peft import LoraConfig, get_peft_model

# Optuna import
import optuna

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

###############################
# Data Loading and Preprocessing
###############################
data = load_data_from_csv('3.7K_python_dataset.csv')
train_df, val_df, test_df = preprocess_data_new(data)

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')

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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

###############################
# Training and Evaluation Functions
###############################
def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
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
        for batch in data_loader:
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

###############################
# Optuna Objective Function
###############################
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 15e-5, 20e-4, log=True)
    lora_r = trial.suggest_int("lora_r", 4, 32, step=4)
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2, step=0.1)

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model.to(device)

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Train for a small number of epochs or steps to quickly evaluate performance
    # (This is for hyperparameter search; in a real run, you might do more epochs)
    for epoch in range(1):  # Just 1 epoch for speed - adjust as needed
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device)
        # We can prune bad trials early if using early_stopping
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss

###############################
# Run Optuna Study
###############################
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # Adjust n_trials as needed

print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

###############################
# Retrain with Best Hyperparameters
###############################
best_learning_rate = best_trial.params['learning_rate']
best_lora_r = best_trial.params['lora_r']
best_lora_alpha = best_trial.params['lora_alpha']
best_lora_dropout = best_trial.params['lora_dropout']

print("\nRetraining with best hyperparameters...\n")
base_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base').to(device)
lora_config = LoraConfig(
    r=best_lora_r,
    lora_alpha=best_lora_alpha,
    lora_dropout=best_lora_dropout,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(base_model, lora_config)
model.to(device)

optimizer = AdamW(model.parameters(), lr=best_learning_rate)

# Now do a full training with chosen best params (adjust epochs as needed)
epochs = 5
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = eval_epoch(model, val_loader, device)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")

# Save final model
model.save_pretrained('./finetuned_lora_model_best')
tokenizer.save_pretrained('./finetuned_lora_model_best')