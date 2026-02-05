import sys
sys.path.append('..')
import math
import torch
import torch.nn as nn

# Transformer example: https://github.com/pytorch/examples/blob/master/word_language_model/model.py


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 601):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class MVFTT(nn.Module):
    """ Multi-View Flight Trajectory Transformer """
    def __init__(self, ninput_stru, nhidden_stru, nhead_stru, nlayer_stru, ninput_sem, nhidden_sem, nhead_sem, nlayer_sem, attn_dropout, pos_droput):
        super(MVFTT, self).__init__()
        self.ninput_stru = ninput_stru
        self.nhidden_stru = nhidden_stru
        self.nhead_stru = nhead_stru
        self.stru_pos_encoder = PositionalEncoding(ninput_stru, pos_droput)

        self.ninput_sem = ninput_sem
        self.nhidden_sem = nhidden_sem
        self.nhead_sem = nhead_sem
        self.sem_pos_encoder = PositionalEncoding(ninput_sem, pos_droput)

        self.Linear_sem = nn.Linear(in_features=self.ninput_sem, out_features=self.ninput_stru)
        self.Linear_traj = nn.Linear(in_features=self.ninput_stru, out_features=self.ninput_stru * 2)

        structural_attn_layers = nn.TransformerEncoderLayer(ninput_stru, nhead_stru, nhidden_stru, attn_dropout)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer_stru)

        semantic_attn_layers = nn.TransformerEncoderLayer(ninput_sem, nhead_sem, nhidden_sem, attn_dropout)
        self.semantic_attn = nn.TransformerEncoder(semantic_attn_layers, nlayer_sem)

    def forward(self, src, attn_mask, src_padding_mask, src_len, srcsemantic):
        src = self.stru_pos_encoder(src)
        structural_rtn = self.structural_attn(src, attn_mask, src_padding_mask)

        srcsemantic = self.sem_pos_encoder(srcsemantic)
        semantic_rtn = self.semantic_attn(srcsemantic, attn_mask, src_padding_mask)
        semantic_rtn = self.Linear_sem(semantic_rtn)

        rtn = torch.cat((structural_rtn, semantic_rtn), dim=-1)

        mask = 1 - src_padding_mask.T.unsqueeze(-1).expand(rtn.shape).float()
        rtn = torch.sum(mask * rtn, 0)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        return rtn  # return traj embeddings
