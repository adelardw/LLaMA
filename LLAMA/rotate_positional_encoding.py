import torch
import torch.nn as nn



class RotatePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        positional_encoding = torch.arange(self.max_seq_len, dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(0, self.embedding_dim,2, dtype=torch.float32)
        div_term = torch.exp(-indices * torch.log(torch.tensor(10000.0)) / self.embedding_dim)


        pe = torch.zeros(self.max_seq_len, self.embedding_dim)
        pe[:, 0::2] = torch.sin(positional_encoding * div_term)
        pe[:, 1::2] = torch.cos(positional_encoding * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        seq_len = x.size(1)
        
        pe = self.pe[:, :seq_len]  
        
        x1 = x[:, :, ::2]  
        x2 = x[:, :, 1::2]  
        
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, ::2] = x1 * pe[:, :, ::2] - x2 * pe[:, :, 1::2]
        x_rotated[:, :, 1::2] = x1 * pe[:, :, 1::2] + x2 * pe[:, :, ::2]

        return x_rotated
    
