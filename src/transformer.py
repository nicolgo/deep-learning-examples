import os
import numpy as np
import random
import math
import json
import pytorch_lightning as pl
import urllib.request
from urllib.error import HTTPError
## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim


def get_pretrained_model():
    # Github URL where saved models are stored for this tutorial
    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial6/"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "../saved_models/tutorial6"
    # Files to download
    pretrained_files = ["ReverseTask.ckpt", "SetAnomalyTask.ckpt"]

    # Create checkpoint path if it doesn't exist yet
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # For each file, check whether it already exists. If not, try downloading it.
    for file_name in pretrained_files:
        file_path = os.path.join(CHECKPOINT_PATH, file_name)
        if "/" in file_name:
            os.makedirs(file_path.rsplit("/",1)[0], exist_ok=True)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Downloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)


def scaled_dot_product(q,k,v,mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q,k.transpose(-2,-1))
    attn_logits = attn_logits/math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0,-9e15)
    attention = F.softmax(attn_logits,dim=-1)
    values = torch.matmul(attention,v)
    return values, attention


def test_attention():
    seq_len, d_k = 3,2
    pl.seed_everything(42)
    q = torch.randn(seq_len,d_k)
    k = torch.randn(seq_len,d_k)
    v = torch.randn(seq_len,d_k)
    values, attention = scaled_dot_product(q,k,v)
    print(f"Q:{q}\n, K:{k}\n,V:{v}\n, Values:{values}\n,Attention:{attention}\n")


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim,embed_dim,num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.qkv_proj = nn.Linear(input_dim,3*embed_dim)
        self.o_proj = nn.Linear(embed_dim,embed_dim)

        self._reset_parameters()
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self,x,mask=None,return_attention=False):
        batch_size,seq_length,_=x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size,seq_length,self.num_heads,3*self.head_dim)
        qkv = qkv.permute(0,2,1,3) # [Batch, Head, SeqLen, Dims]
        q,k,v = qkv.chunk(3,dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q,k,v,mask=mask)
        values = values.permute(0,2,1,3)# [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size,seq_length,self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        

        

if __name__=="__main__":
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "../data"
    
    # Setting the seed
    pl.seed_everything(42)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    test_attention()
    pass

    