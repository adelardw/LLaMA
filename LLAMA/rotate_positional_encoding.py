import torch
import torch.nn as nn

class RotatePositionEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.seq_len =  seq_len

        pos = torch.arange(self.seq_len)[:, None]
        theta = 1.0 / (torch.tensor([10000.])**((torch.arange(0, self.embedding_dim, 2) ) / self.embedding_dim))
        
        pe = pos * theta

        self.register_buffer('pe', pe)
    
    def forward(self, x):

        cos = torch.cos(self.pe)
        sin = torch.sin(self.pe)
        even = x[:, :, ::2]
        odd = x[:, :, 1::2]
        
        position_one = even* cos - odd*sin
        position_two =  odd* cos + even*sin


        rotate_position_encoding = torch.cat([position_one, position_two], dim=-1)

        return rotate_position_encoding
    
