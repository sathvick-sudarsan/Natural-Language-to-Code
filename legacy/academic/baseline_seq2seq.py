# Filename: baseline_seq2seq.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_data, preprocess_data, load_data_from_csv
from nltk.tokenize import word_tokenize
import nltk
import pickle

# Download necessary NLTK data
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess data
data = load_data_from_csv('mbpp_conala.csv')

train_df, val_df, test_df = preprocess_data(data)

# Build vocabulary
def build_vocab(sentences, freq_threshold, special_tokens=None):
    if special_tokens is None:
        special_tokens = []
    word_counts = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Initialize vocabulary with special tokens
    vocab = {}
    idx = 0
    for token in special_tokens:
        vocab[token] = idx
        idx += 1

    # Assign indices to frequent words
    for word, count in word_counts.items():
        if count >= freq_threshold:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

freq_threshold = 1
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

input_sentences = train_df['intent'].tolist()
output_sentences = train_df['snippet'].tolist()

input_vocab = build_vocab(input_sentences, freq_threshold, special_tokens)
output_vocab = build_vocab(output_sentences, freq_threshold, special_tokens)

input_dim = len(input_vocab)
output_dim = len(output_vocab)

# Save vocabularies for use in the app
with open('input_vocab.pkl', 'wb') as f:
    pickle.dump(input_vocab, f)

with open('output_vocab.pkl', 'wb') as f:
    pickle.dump(output_vocab, f)

# Define the Dataset class
class Seq2SeqDataset(Dataset):
    def __init__(self, df, input_vocab, output_vocab, max_len=50):
        self.df = df
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def numericalize(self, text, vocab):
        tokens = word_tokenize(text.lower())
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
        }

# Create datasets and dataloaders
train_dataset = Seq2SeqDataset(train_df, input_vocab, output_vocab)
val_dataset = Seq2SeqDataset(val_df, input_vocab, output_vocab)
test_dataset = Seq2SeqDataset(test_df, input_vocab, output_vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# Loading the test data set 
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Attention mechanism
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, dec_hid_dim]

        # Compute attention
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        # Apply mask (if necessary)
        attention = attention.masked_fill(mask == 0, -1e10)

        return nn.functional.softmax(attention, dim=1)

# Define Encoder, Decoder, Seq2Seq classes with Attention
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [batch_size, src_len, enc_hid_dim * 2]
        # hidden: [num_layers * num_directions, batch_size, enc_hid_dim]

        # Concatenate the final forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))  # [batch_size, dec_hid_dim]
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
        # input: [batch_size]
        # hidden: [batch_size, dec_hid_dim]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]

        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Calculate attention weights
        attn_weights = self.attention(hidden, encoder_outputs, mask)  # [batch_size, src_len]

        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, enc_hid_dim * 2]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + enc_hid_dim * 2]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))  # output: [batch_size, 1, dec_hid_dim]
        hidden = hidden.squeeze(0)  # [batch_size, dec_hid_dim]

        output = output.squeeze(1)  # [batch_size, dec_hid_dim]
        context = context.squeeze(1)  # [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(1)  # [batch_size, emb_dim]

        output = self.fc_out(torch.cat((output, context, embedded), dim=1))  # [batch_size, output_dim]

        return output, hidden, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        mask = (src != input_vocab['<PAD>']).to(self.device)
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)
        mask = self.create_mask(src)

        input = trg[:, 0]  # <SOS> token

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if np.random.rand() < teacher_forcing_ratio else top1

        return outputs

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc_emb_dim = 256
dec_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
dropout = 0.5

attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout).to(device)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=output_vocab['<PAD>'])

# Training loop
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0

    for batch in data_loader:
        src = batch['source_seq'].to(device)
        trg = batch['target_seq'].to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def eval_epoch(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            src = batch['source_seq'].to(device)
            trg = batch['target_seq'].to(device)

            output = model(src, trg, teacher_forcing_ratio=0)

            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

# Start training
epochs = 10  # Increase the number of epochs for better training
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = eval_epoch(model, val_loader, criterion, device)

    

    print(f'Train Loss: {train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')

# Evaluvating the test dataset
test_loss = eval_epoch(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'seq2seq_model.pth')







