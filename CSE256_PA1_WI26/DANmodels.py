import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, embeddings=None):
        self.examples = read_sentiment_examples(infile)
        self.embeddings = embeddings

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        indices = []
        # Convert words to indices
        for w in ex.words:
            idx = self.embeddings.word_indexer.index_of(w)
            if idx == -1:
                indices.append(1) # UNK index
            else:
                indices.append(idx)
        if len(indices) == 0: indices.append(1) # Handle empty lines
        return torch.tensor(indices, dtype=torch.long), torch.tensor(ex.label, dtype=torch.long)
    
class SentimentDatasetBPE(Dataset):
    def __init__(self, infile, tokenizer):
        self.examples = read_sentiment_examples(infile)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ' '.join(ex.words)
        
        # Tokenize using the BPE tokenizer
        # It already returns IDs, so we just wrap in tensor
        indices = self.tokenizer.encode(text)
        
        if len(indices) == 0:
            indices.append(1)  # UNK, not PAD
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(ex.label, dtype=torch.long)
    

def collate_fn_pad(batch):
    data, labels = zip(*batch)
    # Pad sequences with 0 (PAD token)
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return padded_data, torch.tensor(labels)


class DAN(nn.Module):
    def __init__(self, embeddings, hidden_size=300, num_classes=2, dropout_prob=0.3, use_glove=True, embed_dim=300):
        super().__init__()
        print(f"Initializing DAN: use_glove={use_glove}, embed_dim={embed_dim}, hidden_size={hidden_size}, dropout_prob={dropout_prob}")
        if use_glove:
            # Part 1a: Use Pre-trained GloVe
            self.embedding = embeddings.get_initialized_embedding_layer(frozen=False)
            embed_dim = embeddings.get_embedding_length()
        else:
            if embeddings is not None:
                vocab_size = len(embeddings.word_indexer)
                embed_dim = embeddings.get_embedding_length()
            elif vocab_size is None:
                 # Safety fallback
                vocab_size = 10000
            
            # FIX 2: Explicitly cast to int to prevent TypeError
            vocab_size = int(vocab_size)
            embed_dim = int(embed_dim)
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # The classifier network
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = x.long()

        # Masking: Identify which tokens are NOT padding (0)
        mask = (x != 0).float().unsqueeze(-1)

        # Embed the words
        embeds = self.embedding(x)

        # Compute Average (ignoring padding)
        sum_embeds = (embeds * mask).sum(dim=1)
        lengths = mask.sum(dim=1)
        avg_embeds = sum_embeds / lengths.clamp(min=1) # Avoid division by zero
        avg_embeds = self.dropout(avg_embeds)
        # Classification
        out = self.fc1(avg_embeds)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.log_softmax(out)
    


class SubwordDAN(nn.Module):
    def __init__(self, hidden_size=256,num_classes = 2, dropout_prob=0.4,vocab_size = 300, embed_dim=100):
        super().__init__()
        print(f"Initializing SubwordDAN: vocab_size={vocab_size}, embed_dim={embed_dim}, hidden_size={hidden_size}, dropout_prob={dropout_prob}")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        emb = self.embedding(x)
        # add avg pooling
        avg = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Classification
        h = self.fc1(self.dropout(avg))
        h = torch.relu(h)
        h = self.dropout(h)
        return self.log_softmax(self.fc2(h))