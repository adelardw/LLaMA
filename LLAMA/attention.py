import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    def __init__(self, in_size, query_in_size, num_heads, head_size, out_size, query_num_groups=None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.query_in_size = in_size if query_in_size is None else query_in_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.query_num_groups = query_num_groups


        self.q_matrix = nn.Linear(self.query_in_size, self.num_heads * self.head_size, bias=False)
        self.k_matrix = nn.Linear(self.in_size, self.num_heads * self.head_size, bias=False)
        self.v_matrix = nn.Linear(self.in_size, self.num_heads * self.head_size, bias=False)

        self.out_matrix = nn.Linear(self.num_heads * self.head_size, self.out_size)

    
    def forward(self, query, key, value, mask=None):
        # input query -> (batch size, query seq len, query_in_size)
        # input key, value ->  (batch size, seq len, in_size)
        # for each batch and each head we must calcucate relevance and attention

        batch_size = value.size(0)
        query_seq_len = query.size(1)
        seq_len = key.size(1)

        q =self.q_matrix(query).view((batch_size, query_seq_len, self.num_heads, self.head_size))
        k = self.k_matrix(key).view((batch_size, seq_len, self.num_heads, self.head_size))
        v = self.v_matrix(value).view((batch_size, seq_len, self.num_heads, self.head_size))

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        if self.query_num_groups is not None:
            self.group_size = self.num_heads // self.query_num_groups
            assert self.group_size >= 1, 'num heads must be > query num groups'
            #assert self.num_heads % self.query_num_groups == 0

            q = q.view((batch_size, self.query_num_groups, self.group_size, query_seq_len, self.head_size))
            v = v.view((batch_size, self.query_num_groups, self.group_size, seq_len, self.head_size))
            k = k.view((batch_size, self.query_num_groups, self.group_size, seq_len, self.head_size))

        relevance = q @ k.transpose(-2, -1) /  math.sqrt(self.head_size)
        if mask is not None:
                relevance = relevance.masked_fill(mask, -1e20)

        if self.query_num_groups is not None:
            head_attention_i = (F.softmax(relevance, dim=-1) @ v).view(batch_size, self.num_heads, query_seq_len, self.head_size)
        else:
            head_attention_i = (F.softmax(relevance, dim=-1) @ v)
        
        cat_attention = head_attention_i.transpose(1, 2).reshape(batch_size, query_seq_len, self.num_heads * self.head_size)
        return self.out_matrix(cat_attention)
    


