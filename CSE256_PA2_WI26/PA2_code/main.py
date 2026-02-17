import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import argparse
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import SpeechClassifier,LanguageModel
from utilities import Utilities
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


def run_sanity_check(model, tokenizer, part_name):
    print(f"\n[{part_name}] Running Sanity Check (Attention Maps)...")
    model.eval() # Switch to eval mode to get maps
    
    utils = Utilities(tokenizer, model)
    
    # Sentence 1
    # s1 = "The economy is booming and we are doing great."
    # utils.sanity_check(s1, block_size)
    # print(f" -> Generated maps for: '{s1}'")

    # Sentence 2
    s2 = "The American people expect their leaders to act with courage and integrity."
    utils.sanity_check(s2, block_size)
    print(f" -> Generated maps for: '{s2}'")
    
    print("Check your folder for the generated .png files.")
    
    model.train() # Switch back to train mode


# PART 1: Speech Classification
def part1(tokenizer):
    print("\n" + "="*40)
    print("      PART 1: SPEECH CLASSIFICATION")
    print("="*40)
    
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

    model = SpeechClassifier(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, n_hidden, n_output)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for epoch in range(epochs_CLS):
        model.train()
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_CLS_loader)
        test_acc = compute_classifier_accuracy(model, test_CLS_loader)
        print(f"Epoch {epoch+1}/{epochs_CLS} | Loss: {avg_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # CALL SANITY CHECK
    # run_sanity_check(model, tokenizer, "part1")


# PART 2: Language Modeling
def part2(tokenizer):
    print("\n" + "="*40)
    print("      PART 2: LANGUAGE MODELING")
    print("="*40)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    lm_model = LanguageModel(tokenizer.vocab_size, n_embd, n_head, n_layer, block_size, n_hidden)
    lm_model = lm_model.to(device)
    lm_optimizer = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

    print("Starting Language Model training...")
    total_params = sum(p.numel() for p in lm_model.parameters())
    print(f"Decoder Parameters: {total_params}")

    lm_model.train()
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        
        xb, yb = xb.to(device), yb.to(device)
        loss = lm_model(xb, yb) 
        
        lm_optimizer.zero_grad()
        loss.backward()
        lm_optimizer.step()
        
        if i % eval_interval == 0 or i == max_iters - 1:
            perplexity = torch.exp(loss).item()
            print(f"Iter {i}: Train Loss {loss.item():.4f} | Train Perplexity {perplexity:.4f}")

    print("\n--- Evaluating on Test Sets ---")
    test_files = {
        "Obama": "speechesdataset/test_LM_obama.txt",
        "WBush": "speechesdataset/test_LM_wbush.txt",
        "GHBush": "speechesdataset/test_LM_hbush.txt"
    }
    
    for name, filepath in test_files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        perp = compute_perplexity(lm_model, test_loader, eval_iters)
        print(f"{name} Perplexity: {perp:.2f}")
        
    # CALL SANITY CHECK
    # run_sanity_check(lm_model, tokenizer, "part2")


# PART 3: Architectural Exploration
def part3(tokenizer):
    print("\n" + "="*40)
    print("      PART 3: ARCHITECTURAL EXPLORATION")
    print("="*40)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    test_files = {
        "Obama": "speechesdataset/test_LM_obama.txt",
        "WBush": "speechesdataset/test_LM_wbush.txt",
        "GHBush": "speechesdataset/test_LM_hbush.txt"
    }

    # Experimental Configurations
    configs = [
        {"name": "Scaling (n_embd=128)", "window": None, "use_alibi": False, "use_disentangled": False},
        {"name": "AliBi Position",      "window": None, "use_alibi": True,  "use_disentangled": False},
        {"name": "Sparse Window (8)",   "window": 8,    "use_alibi": False, "use_disentangled": False},
        {"name": "Disentangled Attn",   "window": None, "use_alibi": False, "use_disentangled": True},
    ]

    results_table = []

    for cfg in configs:
        print(f"\n>>> Starting Experiment: {cfg['name']}")
        
        model = LanguageModel(
            tokenizer.vocab_size, n_embd=128, n_head=4, n_layer=n_layer, 
            block_size=block_size, n_hidden=n_hidden, 
            window_size=cfg['window'], 
            use_alibi=cfg['use_alibi'], 
            use_disentangled=cfg['use_disentangled']
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")

        model.train()
        final_train_perp = 0
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters: break
            
            xb, yb = xb.to(device), yb.to(device)
            loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % eval_interval == 0 or i == max_iters - 1:
                final_train_perp = torch.exp(loss).item()
                print(f"Iter {i}: Train Perplexity {final_train_perp:.2f}")

        # Evaluation
        model.eval()
        test_perps = {}
        for name, filepath in test_files.items():
            with open(filepath, 'r', encoding='utf-8') as f:
                test_text = f.read()
            test_ds = LanguageModelingDataset(tokenizer, test_text, block_size)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            test_perps[name] = compute_perplexity(model, test_loader, eval_iters)
        
        results_table.append({
            "Config": cfg['name'],
            "Train Perp": final_train_perp,
            "Obama": test_perps["Obama"],
            "WBush": test_perps["WBush"],
            "GHBush": test_perps["GHBush"]
        })

    print("\n" + "="*85)
    print(f"{'Configuration':<25} | {'Train P.':<10} | {'Obama':<8} | {'WBush':<8} | {'GHBush':<8}")
    print("-" * 85)
    for res in results_table:
        print(f"{res['Config']:<25} | {res['Train Perp']:<10.2f} | {res['Obama']:<8.2f} | {res['WBush']:<8.2f} | {res['GHBush']:<8.2f}")
    print("="*85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSE 256 PA2 Runner")
    parser.add_argument("--part", type=str, required=True, choices=["part1", "part2", "part3"],
                        help="Choose which part of the assignment to run")
    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.part == "part1":
        part1(tokenizer)
    elif args.part == "part2":
        part2(tokenizer)
    elif args.part == "part3":
        part3(tokenizer)

