import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# PART 1: Standard Transformer Implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1, is_causal=False, 
                 window_size=None, use_alibi=False, use_disentangled=False):
        """
            n_embd: embedding dimension
            n_head: number of attention heads
            block_size: maximum sequence length (for masking)
            dropout: dropout rate for attention and output     
            is_causal: whether to apply causal masking
            window_size: if not None, applies a sparse attention pattern with this local window size
            use_alibi: whether to apply ALiBi positional bias instead of standard positional embeddings
            use_disentangled: whether to implement disentangled attention with separate position projections
        """
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.is_causal = is_causal
        self.window_size = window_size
        self.use_alibi = use_alibi
        self.use_disentangled = use_disentangled
        
        # Standard projections
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        
        # Disentangled Position Projections
        
        if self.use_disentangled:
            # separate linear layers to project positional embeddings into key and query spaces
            self.pos_key = nn.Linear(n_embd, n_embd)
            self.pos_query = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        if self.use_alibi:
            slopes = torch.tensor([2**(-(8/n_head) * (i+1)) for i in range(n_head)])
            self.register_buffer("alibi_slopes", slopes.view(n_head, 1, 1))

    def forward(self, x, pos_emb=None):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Content-to-Content Score
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        
        # Disentangled Attention Components
        if self.use_disentangled and pos_emb is not None:
            # pos_emb is shape (B, T, C)
            pk = self.pos_key(pos_emb).view(B, T, self.n_head, self.head_size).transpose(1, 2)
            pq = self.pos_query(pos_emb).view(B, T, self.n_head, self.head_size).transpose(1, 2)
            
            # Content-to-Position + Position-to-Content
            c2p = q @ pk.transpose(-2, -1)
            p2c = pq @ k.transpose(-2, -1)
            wei = (wei + c2p + p2c) * (1.0 / math.sqrt(3 * self.head_size)) # Factor for stability

        # Apply Masks (Causal + Sparse)
        if self.is_causal:
            mask = self.tril[:T, :T]
            if self.window_size is not None:
                # Sparse pattern: window-based local attention
                sparse_mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0)
                sparse_mask = torch.triu(sparse_mask, diagonal=-self.window_size)
                mask = mask * sparse_mask # Combine masks correctly
            
            wei = wei.masked_fill(mask == 0, float('-inf'))

        # ALiBi Bias
        if self.use_alibi:
            # Create distance matrix and apply ALiBi slopes
            dist = torch.arange(T, device=x.device).view(1, -1) - torch.arange(T, device=x.device).view(-1, 1)
            dist = torch.abs(dist).unsqueeze(0)
            wei = wei - (self.alibi_slopes * dist) 

        wei = F.softmax(wei, dim=-1)
        out = (self.attn_dropout(wei) @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out)), wei

class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_hidden, dropout, is_causal, 
                 window_size, use_alibi, use_disentangled):
        super().__init__()
        
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, is_causal, 
                                     window_size, use_alibi, use_disentangled)
        
        self.ffwd = FeedForward(n_embd, n_hidden, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, pos_emb=None):
        attn_out, weights = self.sa(self.ln1(x), pos_emb)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, weights

class SpeechClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_hidden, n_output, dropout=0.1):
        """
            Standard Transformer-based classifier architecture:
            - Token and Position Embeddings
            - Stacked Transformer Blocks (non-causal, no sparse masking)
            - LayerNorm + Mean Pooling
            - MLP Classifier Head       
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size,n_hidden, dropout,is_causal=False, window_size=None, use_alibi=False, use_disentangled=False) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier_head = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_output)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
            x = tok_emb + pos_emb
            
            attn_maps = []
            for block in self.blocks:
                x, weights = block(x) 
                # weights shape: (B, n_head, T, T)

                for h in range(weights.shape[1]):
                    attn_maps.append(weights[:, h, :, :])
                
            x = self.ln_f(x).mean(dim=1) 
            logits = self.classifier_head(x)
            
            # If targets are provided, compute and return loss for training
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
                return logits, loss
                
            if not torch.is_grad_enabled():
                return logits
            
            # Otherwise, return (logits, attn_maps) for sanity check visualization
            return logits, [m.detach() for m in attn_maps]
            

#part2
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_hidden, 
                 dropout=0.1, window_size=None, use_alibi=False, use_disentangled=False):
        """
            Standard Transformer-based language model architecture:
            - Token and Position Embeddings (or ALiBi bias)
            - Stacked Causal Transformer Blocks (with optional sparse masking)
            - LayerNorm + Linear head for next-token prediction
            - Supports ALiBi and Disentangled Attention based on flags"""
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.use_alibi = use_alibi
        self.use_disentangled = use_disentangled
        
        
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, n_hidden, dropout, is_causal=True, 
                  window_size=window_size, use_alibi=use_alibi, use_disentangled=use_disentangled) 
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        
        # Always compute pos_emb for Disentangled or standard positioning
        pos_indices = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_indices).unsqueeze(0).expand(B, -1, -1)
        
        # ALiBi replaces standard positional addition
        x = tok_emb if self.use_alibi else tok_emb + pos_emb
        
        attn_maps = []
        for block in self.blocks:
            # Pass pos_emb if disentangled mode is on
            x, weights = block(x, pos_emb if self.use_disentangled else None)
            if targets is None:
                for h in range(weights.shape[1]):
                    attn_maps.append(weights[:, h, :, :])
        
        logits = self.lm_head(self.ln_f(x))
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
            return loss
        return logits, [m.detach() for m in attn_maps]