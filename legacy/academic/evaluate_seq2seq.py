# Filename: evaluate_seq2seq.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pickle

from torch.utils.data import Dataset, DataLoader
import logging

from utils import load_data_from_csv, preprocess_data
nltk.download('punkt')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

########################################
# Load vocabularies
########################################
with open('input_vocab.pkl', 'rb') as f:
    input_vocab = pickle.load(f)

with open('output_vocab.pkl', 'rb') as f:
    output_vocab = pickle.load(f)

# Invert output_vocab for decoding
inv_output_vocab = {v: k for k, v in output_vocab.items()}

########################################
# Dataset class (same as training)
########################################
class Seq2SeqDataset(Dataset):
    def __init__(self, df, input_vocab, output_vocab, max_len=50):
        self.df = df.reset_index(drop=True)
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def numericalize(self, text, vocab):
        tokens = nltk.word_tokenize(text.lower())
        numericalized = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        return numericalized

    def __getitem__(self, index):
        source_text = self.df.iloc[index]['intent']
        target_text = self.df.iloc[index]['snippet']

        source_seq = self.numericalize(source_text, self.input_vocab)
        target_seq = self.numericalize(target_text, self.output_vocab)

        # Add <SOS> and <EOS> tokens to target_seq
        target_seq = [self.output_vocab['<SOS>']] + target_seq + [self.output_vocab['<EOS>']]

        # Pad sequences
        source_seq = source_seq[:self.max_len]
        target_seq = target_seq[:self.max_len]

        source_seq += [self.input_vocab['<PAD>']] * (self.max_len - len(source_seq))
        target_seq += [self.output_vocab['<PAD>']] * (self.max_len - len(target_seq))

        return {
            'source_seq': torch.tensor(source_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'source_text': source_text,
            'target_text': target_text
        }

########################################
# Model architecture (must match training code)
########################################

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return nn.functional.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        attn_weights = self.attention(hidden, encoder_outputs, mask)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)

        output = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return output, hidden, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, input_vocab, output_vocab):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def create_mask(self, src):
        mask = (src != self.input_vocab['<PAD>']).to(self.device)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)
        input = trg[:, 0]  # <SOS>

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if np.random.rand() < teacher_forcing_ratio else top1

        return outputs

    def generate(self, src, max_len=50):
        # Generate output tokens with greedy decoding
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src)
            mask = self.create_mask(src)

            input_token = torch.tensor([self.output_vocab['<SOS>']]).to(self.device)
            generated_tokens = []

            for _ in range(max_len):
                output, hidden, _ = self.decoder(input_token, hidden, encoder_outputs, mask)
                top1 = output.argmax(1)
                input_token = top1
                token_id = top1.item()
                if token_id == self.output_vocab['<EOS>']:
                    break
                generated_tokens.append(token_id)

            return generated_tokens

########################################
# Main evaluation script
########################################
if __name__ == "__main__":
    # Load data
    data = load_data_from_csv('mbpp_conala.csv')
    _, _, test_df = preprocess_data(data)

    # Hyperparams must match training
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hid_dim = 512
    dec_hid_dim = 512
    dropout = 0.5

    # Rebuild model
    attn = Attention(enc_hid_dim, dec_hid_dim)
    encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout)
    decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Seq2Seq(encoder, decoder, device, input_vocab, output_vocab).to(device)

    # Load trained weights
    model.load_state_dict(torch.load('seq2seq_model.pth', map_location=device))

    # Create test dataset and loader
    test_dataset = Seq2SeqDataset(test_df, input_vocab, output_vocab)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    references = []

    smoothie = SmoothingFunction().method4
    bleu_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []

    print("Evaluating on test set...")

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src = batch['source_seq'].to(device)  # [1, seq_len]
            trg = batch['target_seq'].to(device)  # [1, seq_len]
            target_text = batch['target_text'][0]

            # Generate predictions
            generated_ids = model.generate(src, max_len=50)
            # Decode predicted tokens
            pred_tokens = [inv_output_vocab.get(idx, '<UNK>') for idx in generated_ids]
            # Stop at <PAD> if present
            if '<PAD>' in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index('<PAD>')]
            pred_text = ' '.join(pred_tokens)

            # Decode reference
            ref_tokens = nltk.word_tokenize(target_text.lower())
            pred_tokens_nltk = nltk.word_tokenize(pred_text.lower())

            # BLEU
            bleu_score = sentence_bleu([ref_tokens], pred_tokens_nltk, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)

            # ROUGE
            rouge_scores = scorer.score(target_text, pred_text)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougel_scores.append(rouge_scores['rougeL'].fmeasure)

            predictions.append(pred_text)
            references.append(target_text)

    avg_bleu = np.mean(bleu_scores)
    avg_rouge1 = np.mean(rouge1_scores)
    avg_rouge2 = np.mean(rouge2_scores)
    avg_rougel = np.mean(rougel_scores)

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
        print(f"Reference:\n{references[i]}")
        print(f"Prediction:\n{predictions[i]}\n")
