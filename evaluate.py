# Filename: evaluate.py

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import DataLoader
from transformer_model import CodeDataset
from baseline_seq2seq import Seq2Seq, Encoder, Decoder
from utils import load_data, preprocess_data
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_seq2seq_model(model, dataset, input_vocab, output_vocab):
    model.eval()
    bleu_scores = []
    exact_matches = 0
    total = 0
    inv_output_vocab = {v: k for k, v in output_vocab.items()}

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Seq2Seq"):
            src = batch['source_seq'].to(device)
            trg = batch['target_seq'].to(device)

            output = model(src, trg, teacher_forcing_ratio=0)
            output = output.argmax(2)  # Get the indices of the max probability

            pred_tokens = [inv_output_vocab.get(idx.item(), '<UNK>') for idx in output.squeeze()]
            ref_tokens = [inv_output_vocab.get(idx.item(), '<UNK>') for idx in trg.squeeze()]

            pred_code = ' '.join(pred_tokens).replace('<PAD>', '').replace('<EOS>', '').strip()
            ref_code = ' '.join(ref_tokens).replace('<PAD>', '').replace('<EOS>', '').strip()

            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([ref_code.split()], pred_code.split(), smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

            if pred_code == ref_code:
                exact_matches += 1
            total += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    exact_match_accuracy = exact_matches / total

    print(f'Seq2Seq Model - Average BLEU Score: {avg_bleu:.4f}')
    print(f'Seq2Seq Model - Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%')

def evaluate_transformer_model(model, tokenizer, dataset):
    model.eval()
    bleu_scores = []
    exact_matches = 0
    total = 0

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Transformer"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256
            )

            pred_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ref_code = tokenizer.decode(labels[0], skip_special_tokens=True)

            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([ref_code.split()], pred_code.split(), smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

            if pred_code.strip() == ref_code.strip():
                exact_matches += 1
            total += 1

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    exact_match_accuracy = exact_matches / total

    print(f'Transformer Model - Average BLEU Score: {avg_bleu:.4f}')
    print(f'Transformer Model - Exact Match Accuracy: {exact_match_accuracy * 100:.2f}%')

# Load data and models
data = load_data()
_, _, test_df = preprocess_data(data)

# Seq2Seq Model Evaluation
from baseline_seq2seq import Seq2SeqDataset, input_vocab, output_vocab

test_dataset_seq2seq = Seq2SeqDataset(test_df, input_vocab, output_vocab)

# Load the trained Seq2Seq model
input_dim = len(input_vocab)
output_dim = len(output_vocab)
emb_dim = 256
hid_dim = 512
num_layers = 2
dropout = 0.5

encoder = Encoder(input_dim, emb_dim, hid_dim, num_layers, dropout).to(device)
decoder = Decoder(output_dim, emb_dim, hid_dim, num_layers, dropout).to(device)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)

seq2seq_model.load_state_dict(torch.load('seq2seq_model.pth', map_location=device))

evaluate_seq2seq_model(seq2seq_model, test_dataset_seq2seq, input_vocab, output_vocab)

# Transformer Model Evaluation
from transformer_model import CodeDataset

tokenizer = T5Tokenizer.from_pretrained('./finetuned_model')
transformer_model = T5ForConditionalGeneration.from_pretrained('./finetuned_model').to(device)

test_dataset_transformer = CodeDataset(test_df, tokenizer)
evaluate_transformer_model(transformer_model, tokenizer, test_dataset_transformer)
