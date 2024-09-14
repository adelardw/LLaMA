import torch
import torch.nn as nn
from attention import MultiheadAttention
from collections import OrderedDict
from rotate_positional_encoding import RotatePositionalEncoding



class LlamaBlock(nn.Module):
    def __init__(self, in_size, 
                       num_heads,
                       head_size,
                       out_size,
                       ff_hidden_size,
                       query_num_groups,
                       max_seq_len,
                       dropout_p = 0.5
                       ):
        super().__init__()
        

        self.rms_norm_1 = nn.RMSNorm(in_size)
        self.attention = MultiheadAttention(in_size=in_size,num_heads=num_heads,
                                            head_size=head_size, out_size=out_size,
                                            query_num_groups=query_num_groups,query_in_size=None)
        
        self.pe = RotatePositionalEncoding(max_seq_len=max_seq_len,
                                           embedding_dim=in_size)
        
        self.adapt_residual = nn.Linear(in_size, out_size) if in_size!=out_size else nn.Identity()

        self.rms_norm_2 = nn.RMSNorm(out_size)
        self.dropout = nn.Dropout(dropout_p)

        self.ff_net = nn.Sequential(OrderedDict(
                                    [('ff_1',nn.Linear(out_size, ff_hidden_size)),
                                     ('ff_2',nn.SiLU()),
                                     ('ff_3', nn.Dropout(dropout_p)),
                                     ('ff_4',nn.Linear(ff_hidden_size, out_size))]))
        
    
    def forward(self, x, mask=None):
        
        value = self.rms_norm_1(x)
        query = self.pe(value)
        key = self.pe(value)

        att = self.attention(query, key, value, mask)
        residual = self.adapt_residual(x) + att

        norm = self.dropout(self.rms_norm_2(residual))
        ff_out = self.ff_net(norm)

        return residual + ff_out


