"""Attention Seq2Seq adapted from legacy/academic/baseline_seq2seq.py."""

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.energy = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim, decoder_hidden_dim)
        self.score = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, outputs, mask):
        repeated = hidden.unsqueeze(1).expand(-1, outputs.size(1), -1)
        scores = self.score(torch.tanh(self.energy(torch.cat((repeated, outputs), 2))))
        return torch.softmax(scores.squeeze(2).masked_fill(~mask, float("-inf")), dim=1)


class Encoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config["encoder_embedding_dim"], padding_idx=0)
        self.rnn = nn.GRU(
            config["encoder_embedding_dim"],
            config["encoder_hidden_dim"],
            bidirectional=True,
            batch_first=True,
        )
        self.project = nn.Linear(config["encoder_hidden_dim"] * 2, config["decoder_hidden_dim"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, source, lengths):
        embedded = self.dropout(self.embedding(source))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=source.size(1)
        )
        hidden = torch.tanh(self.project(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        encoder_width = config["encoder_hidden_dim"] * 2
        embedding_width = config["decoder_embedding_dim"]
        hidden_width = config["decoder_hidden_dim"]
        self.embedding = nn.Embedding(vocab_size, embedding_width, padding_idx=0)
        self.attention = Attention(config["encoder_hidden_dim"], hidden_width)
        self.rnn = nn.GRU(embedding_width + encoder_width, hidden_width, batch_first=True)
        self.project = nn.Linear(hidden_width + encoder_width + embedding_width, vocab_size)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, token, hidden, outputs, mask):
        embedded = self.dropout(self.embedding(token)).unsqueeze(1)
        weights = self.attention(hidden, outputs, mask).unsqueeze(1)
        context = torch.bmm(weights, outputs)
        state, next_hidden = self.rnn(torch.cat((embedded, context), dim=2), hidden.unsqueeze(0))
        logits = self.project(
            torch.cat((state.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        )
        return logits, next_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, source_size, target_size, config):
        super().__init__()
        self.encoder = Encoder(source_size, config)
        self.decoder = Decoder(target_size, config)
        self.source_pad = 0
        self.target_pad = 0
        self.target_sos = 2
        self.target_eos = 3

    def forward(self, source, lengths, target, teacher_forcing_ratio, generator):
        outputs, hidden = self.encoder(source, lengths)
        mask = source.ne(self.source_pad)
        token = target[:, 0]
        logits = []
        for step in range(1, target.size(1)):
            score, hidden = self.decoder(token, hidden, outputs, mask)
            logits.append(score)
            use_teacher = torch.rand((), generator=generator).item() < teacher_forcing_ratio
            token = target[:, step] if use_teacher else score.argmax(1)
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, source, lengths, max_tokens):
        outputs, hidden = self.encoder(source, lengths)
        mask = source.ne(self.source_pad)
        token = torch.full((source.size(0),), self.target_sos, device=source.device)
        generated = [[] for _ in range(source.size(0))]
        finished = [False] * source.size(0)
        for _ in range(max_tokens):
            score, hidden = self.decoder(token, hidden, outputs, mask)
            token = score.argmax(1)
            for index, value in enumerate(token.tolist()):
                if not finished[index]:
                    if value == self.target_eos:
                        finished[index] = True
                    else:
                        generated[index].append(value)
            if all(finished):
                break
        return generated
