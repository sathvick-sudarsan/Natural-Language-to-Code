import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from peft import LoraConfig, get_peft_model, TaskType
import logging
from utils import load_data, preprocess_data, preprocess_data_new
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

# Suppress tokenizer warnings
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Load and preprocess data
data = load_data('/content/mbpp_conala.csv')
train_df, val_df, test_df = preprocess_data(data)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')

# Dataset Class
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

        labels = target_encoding['input_ids'].squeeze().long()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels,
        }

# Training Function
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

# Evaluation Function
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

            # Generate predictions for inspection
            generated_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256)
            batch_predictions = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            batch_references = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

            predictions.extend(batch_predictions)
            references.extend(batch_references)

    # Print sample outputs
    for i in range(min(3, len(predictions))):
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 50)

    return total_loss / len(data_loader), predictions, references

# Objective Function for Optuna
def objective(trial):
    # Hyperparameter search space
    lora_r = trial.suggest_int('lora_r', 8, 64, step=8)
    lora_alpha = trial.suggest_int('lora_alpha', 16, 128, step=16)
    lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.3, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=False
    )

    # Load model and apply LoRA
    base_model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5-base')
    model = get_peft_model(base_model, lora_config)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = CodeDataset(train_df, tokenizer)
    val_dataset = CodeDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training and evaluation loop
    best_val_loss = float('inf')
    for epoch in range(3):  # Use fewer epochs for faster tuning
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = eval_epoch(model, val_loader, tokenizer, device)

        # Early stopping if no improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            break

    return best_val_loss

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Print best hyperparameters
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

