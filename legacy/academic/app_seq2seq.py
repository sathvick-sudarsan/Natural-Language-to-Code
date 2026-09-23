# Filename: app_seq2seq.py

import streamlit as st
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
import pickle

# Import necessary modules from your project
from baseline_seq2seq import Seq2Seq, Encoder, Decoder, Attention
import numpy as np

# Download NLTK data
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load the input and output vocabularies
def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

input_vocab = load_vocab('input_vocab.pkl')
output_vocab = load_vocab('output_vocab.pkl')

# Reverse output_vocab to get index to word mapping
inv_output_vocab = {v: k for k, v in output_vocab.items()}

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model architecture
input_dim = len(input_vocab)
output_dim = len(output_vocab)

enc_emb_dim = 256
dec_emb_dim = 256
enc_hid_dim = 512
dec_hid_dim = 512
dropout = 0.5

attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, dropout).to(device)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dropout, attn).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load the trained model parameters
model.load_state_dict(torch.load('seq2seq_model.pth', map_location=device))
model.eval()

# Function to preprocess and numericalize input text
def preprocess_input(text, vocab, max_len=50):
    tokens = word_tokenize(text.lower())
    numericalized = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    numericalized = numericalized[:max_len]
    numericalized += [vocab['<PAD>']] * (max_len - len(numericalized))
    return torch.tensor(numericalized, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension

# Function to generate code from input text
def generate_code(model, input_text, input_vocab, output_vocab, inv_output_vocab, max_len=50):
    model.eval()
    with torch.no_grad():
        src_tensor = preprocess_input(input_text, input_vocab, max_len)
        encoder_outputs, hidden = model.encoder(src_tensor)
        mask = model.create_mask(src_tensor)

        trg_indexes = [output_vocab['<SOS>']]

        for i in range(max_len):
            trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == output_vocab['<EOS>']:
                break

        trg_tokens = [inv_output_vocab.get(idx, '<UNK>') for idx in trg_indexes[1:-1]]  # Exclude <SOS> and <EOS>
        return ' '.join(trg_tokens)

# Streamlit application
st.title('Seq2Seq Model with Attention: Natural Language to Python Code Generator')

user_input = st.text_area('Enter a programming task description in English:', height=150)

if st.button('Generate Code with Seq2Seq'):
    if user_input.strip():
        generated_code = generate_code(
            model, user_input, input_vocab, output_vocab, inv_output_vocab
        )
        st.subheader('Generated Code (Seq2Seq with Attention):')
        st.code(generated_code, language='python')
    else:
        st.warning('Please enter a description.')
