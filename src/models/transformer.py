import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from config import cfg
from modules import Scaler


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(cfg['bptt'], embedding_size)

    def forward(self, x):
        S, N = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S)).t()
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = nn.Embedding(num_tokens + 1, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) + self.positional_embedding(src)
        src = self.dropout(self.norm(src))
        return src


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, rate):
        super().__init__()
        self.mha = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.init_param()

    def init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        return

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output = self.mha(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size, eps=1e-12)
        self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.linear1(src))))
        return out


class Transformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, rate):
        super().__init__()
        self.num_tokens = num_tokens
        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout, rate)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = Decoder(num_tokens, embedding_size)

    def forward(self, input):
        output = {}
        src = input['label'].clone().transpose(0, 1)
        S, N = src.size()
        d = torch.distributions.bernoulli.Bernoulli(probs=cfg['mask_rate'])
        mask = d.sample((S, N))
        src[mask == 1] = self.num_tokens
        src = src.detach()
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src)
        out = self.decoder(src)
        out = out.permute(1, 2, 0)
        if 'label_split' in input:
            label_mask = torch.zeros((cfg['num_tokens'], 1), device=out.device)
            label_mask[input['label_split']] = 1
            out = out * label_mask
        output['score'] = out
        output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def transformer(model_rate=1):
    num_tokens = cfg['num_tokens']
    embedding_size = int(np.ceil(model_rate * cfg['transformer']['embedding_size']))
    num_heads = cfg['transformer']['num_heads']
    hidden_size = int(np.ceil(model_rate * cfg['transformer']['hidden_size']))
    num_layers = cfg['transformer']['num_layers']
    dropout = cfg['transformer']['dropout']
    scaler_rate = model_rate / cfg['global_model_rate']
    model = Transformer(num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout, scaler_rate)
    return model