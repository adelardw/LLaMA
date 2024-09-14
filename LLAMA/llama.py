
import torch.nn as nn
from llama_block import LlamaBlock


class LLama(nn.Module):
    def __init__(self,vocab_size, embedding_dim, num_heads, 
                       head_size,out_size,ff_hidden_size,
                       max_seq_len, query_num_groups,
                       num_blocks, dropout_p = 0.5, pad_idx=None):
        
        super().__init__()


        self.embedded_layer = nn.Embedding(vocab_size, embedding_dim,
                                           padding_idx = pad_idx)

        self.neck = nn.ModuleDict({
            f'llama_block_{i}': LlamaBlock(in_size = embedding_dim if i == 0 else out_size, 
                                            num_heads = num_heads,
                                            head_size = head_size,
                                            out_size = out_size,
                                            ff_hidden_size = ff_hidden_size,
                                            query_num_groups = query_num_groups,
                                            max_seq_len = max_seq_len,
                                            dropout_p = dropout_p) for i in range(num_blocks)})
        
        self.norm = nn.RMSNorm(out_size)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(out_size, vocab_size)
    
    def forward(self, x, mask=None):

        out = self.embedded_layer(x)

        for block in self.neck.values():
            out = block(out, mask)
        
        norm = self.dropout(self.norm(out))

        return self.linear(norm)