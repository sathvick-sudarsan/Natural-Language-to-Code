import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import load_data, preprocess_data, load_data_from_csv
from nltk.tokenize import word_tokenize
import nltk
import pickle


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

def load_glove_embeddings(glove_file, vocab, embedding_dim=100):
    embeddings = np.zeros((len(vocab), embedding_dim))  # Initialize embedding matrix
    word_to_idx = vocab  # Use the vocab you created in the original code

    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')

            # If word exists in vocab, assign its GloVe embedding
            if word in word_to_idx:
                embeddings[word_to_idx[word]] = vector
    return embeddings


x = {'text', 'the', 'leader', 'prime minister',
     'natural', 'language'}

special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
input = build_vocab(x,1,special_tokens)
#print(input)

glove_file = '/Users/vineeth/Desktop/Rutgers/Semester 3/NLP/Project/glove/glove.6B.300d.txt'  # Path to GloVe file
glove_embeddings = load_glove_embeddings(glove_file, input, embedding_dim=300)
print(glove_embeddings)
print("Embedding Matrix Shape:", glove_embeddings.shape)

word = 'language'  # Replace with the word you want to inspect
word_idx = input.get(word, input['<UNK>'])  # Get index, default to <UNK>
print(f"Embedding for '{word}':\n", glove_embeddings[word_idx])

print("Embedding for Lnaguage:", glove_embeddings[input['language']])
#print("Embedding for <UNK>:", glove_embeddings[vocab['<UNK>']])
