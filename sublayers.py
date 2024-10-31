import torch.nn as nn
from multiHeadAttention import MultiHeadAttention


class Encoder_Layer(nn.Module):
    """
    Encoder Block
    
    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    dropout:
        The dropout value
    """
    
    def __init__(self,
                 d_model,
                 nhead,
                 dropout,
                 ffn_dim=64):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask):
        
        # Attention
        residual = q
        x = self.Attention(query=q, key=kv, value=kv, mask=mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm2(x + residual)
        
        return x
    

class Decoder_Layer(nn.Module):
    """
    Decoder Block
    
    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    dropout:
        The dropout value
    """
    
    def __init__(self,
                 d_model,
                 nhead,
                 dropout,
                 ffn_dim=64):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.layerNorm3 = nn.LayerNorm(d_model)
    def forward(self, m,src_mask,q,trg_mask):
        # self-attention
        residual = q
        x = self.Attention(query=q, key=q, value=q, mask=trg_mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        #cross-attention
        residual = x
        x = self.Attention(query=x, key=m, value=m, mask=src_mask)
        x = self.dropout(x)
        x = self.layerNorm2(x + residual)

        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm3(x + residual)
        
        return x