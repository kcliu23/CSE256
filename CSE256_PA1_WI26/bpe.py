import re
from collections import Counter, defaultdict

class BPETokenizer:
    def __init__(self):
        self.merges = []              # ordered list of merge pairs
        self.merge_map = {}           # (a,b) -> ab
        self.vocab = set()
        self.token_to_id = {}
        self.cache = {}

    def get_stats(self, vocab):
        # Get counts of all symbol pairs in the vocabulary
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        # Merge all occurrences of the given pair in the vocabulary
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        merged = ''.join(pair)
        for word in v_in:
            v_out[p.sub(merged, word)] = v_in[word]
        return v_out

    def train(self, filepath, vocab_size=3000):
        print(f"--- Training BPE (target vocab = {vocab_size}) ---")
        # Build initial vocabulary from training data
        word_counts = Counter()
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')[-1]
                for w in text.split():
                    chars = ' '.join(list(w)) + ' </w>'
                    word_counts[chars] += 1

        self.vocab = set()
        for w in word_counts:
            self.vocab.update(w.split())
        # Build BPE merges
        while len(self.vocab) < vocab_size:
            pairs = self.get_stats(word_counts)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            merged = ''.join(best)
            self.merge_map[best] = merged
            self.vocab.add(merged)

            word_counts = self.merge_vocab(best, word_counts)

            if len(self.merges) % 100 == 0:
                print(f"Merges: {len(self.merges)}, vocab: {len(self.vocab)}")

        self.token_to_id = {"<PAD>": 0, "<UNK>": 1}
        for i, tok in enumerate(sorted(self.vocab)):
            self.token_to_id[tok] = i + 2

        print(f"BPE complete. Final vocab size: {len(self.token_to_id)}")

    def encode(self, text):
        encoded = []
        # Tokenize each word
        for word in text.strip().split():
            if word in self.cache:
                tokens = self.cache[word]
            else:
                tokens = list(word) + ['</w>']
                for pair in self.merges:
                    i = 0
                    new_tokens = []
                    # Merge all occurrences of the current pair
                    while i < len(tokens):
                        if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == pair:
                            new_tokens.append(self.merge_map[pair])
                            i += 2
                        else:
                            new_tokens.append(tokens[i])
                            i += 1
                    tokens = new_tokens
                # Cache the result
                tokens = [t for t in tokens if t != '</w>']
                self.cache[word] = tokens

            for t in tokens:
                encoded.append(self.token_to_id.get(t, 1))

        return encoded