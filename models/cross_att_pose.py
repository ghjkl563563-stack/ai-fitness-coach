import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, kv):
        attn_out, _ = self.mha(query=x, key=kv, value=kv)
        x = self.norm1(x + torch.tanh(self.alpha) * self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x, _

class PoseCrossAttModel(nn.Module):
    def __init__(self, pose_dim=256, aux_dim=1, hidden=128, aux_hidden=128, num_layers=2, num_heads=4, num_classes=5, num_errors=5):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, hidden)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(aux_hidden, hidden)
        )
        self.pos_enc = PositionalEncoding(hidden)
        self.layers = nn.ModuleList([GatedCrossAttentionBlock(dim=hidden, num_heads=num_heads) for _ in range(num_layers)])
        self.temporal_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads, batch_first=True), num_layers=1)

        self.cls_head = nn.Linear(hidden, num_classes)
        self.score_head = nn.Linear(hidden, 1)
        self.error_head = nn.Linear(hidden, num_errors)

    def forward(self, pose, aux=None, exemplar=None):
        # 使用者動作特徵 (Query)
        x = self.pose_proj(pose)
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.pos_enc(x)

        # 決定 Key/Value (範本優先，生理數據其次)
        if exemplar is not None:
            kv = self.pose_proj(exemplar)
            if kv.dim() == 2: kv = kv.unsqueeze(1)
            kv = self.pos_enc(kv)
        elif aux is not None:
            kv = self.aux_mlp(aux)
            if kv.dim() == 2: kv = kv.unsqueeze(1)
            kv = self.pos_enc(kv)
        else:
            kv = x

        for layer in self.layers:
            x, _ = layer(x, kv)

        x = self.temporal_enc(x)
        feat = x[:, -1, :]
        return self.cls_head(feat), self.score_head(feat), self.error_head(feat), None
