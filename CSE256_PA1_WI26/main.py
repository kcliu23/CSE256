import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import SentimentDatasetDAN, SentimentDatasetBPE, DAN,SubwordDAN, collate_fn_pad
from sentiment_data import read_word_embeddings
from bpe import BPETokenizer
import os
import copy


# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        # X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def run_experiment(model, train_loader, test_loader, lr, weight_decay=0.0, patience=10):
    loss_fn = nn.NLLLoss()
    
    # Initialize Optimizer with the specific LR and Weight Decay passed in
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"Starting training with LR={lr}, WD={weight_decay}, Patience={patience}")
    
    all_train_acc, all_dev_acc = [], []
    all_train_loss, all_dev_loss = [], []
    
    best_dev_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0 
    
    for epoch in range(100):
        # Train
        train_acc, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_acc.append(train_acc)
        all_train_loss.append(train_loss)

        # Eval
        dev_acc, dev_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        
        all_dev_acc.append(dev_acc)
        all_dev_loss.append(dev_loss)

        # Early Stopping Logic
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Epoch {epoch+1}: Train {train_acc:.3f} | Dev {dev_acc:.3f} | Loss {train_loss:.3f} *NEW PEAK*")
        else:
            patience_counter += 1
            if epoch % 10 == 9:
                print(f"Epoch {epoch+1}: Train {train_acc:.3f} | Dev {dev_acc:.3f} | Loss {train_loss:.3f}")

        if patience_counter >= patience:
            print(f"Early Stopping at Epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)
    print(f"Final Best Dev Accuracy: {best_dev_acc:.3f}\n")
    
    return all_train_acc, all_dev_acc, all_train_loss, all_dev_loss

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt", vectorizer=train_data.vectorizer, train=False)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file =os.path.join(results_dir, 'train_accuracy.png')
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = os.path.join(results_dir, 'dev_accuracy.png')
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # Load Embeddings
        print("Loading embeddings...")
        embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt") 

        # Load Data
        print("Loading data...")
        train_data = SentimentDatasetDAN("data/train.txt", embeddings=embeddings)
        dev_data = SentimentDatasetDAN("data/dev.txt", embeddings=embeddings)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn_pad)
        test_loader = DataLoader(dev_data, batch_size=64, shuffle=False, collate_fn=collate_fn_pad)

        # GloVe Initialization 
        print("\n--- Running Experiment 1: GloVe Initialization ---")
        model_glove = DAN(embeddings, hidden_size=300, dropout_prob=0.3, use_glove=True)
        glove_train_acc, glove_dev_acc, glove_train_loss, glove_dev_loss = run_experiment(
            model_glove, train_loader, test_loader, 
            lr=0.0001,          
            weight_decay=1e-5, 
            patience=15
        )

        # Random Initialization
        print("\n--- Running Experiment 2: Random Initialization ---")
        model_rand = DAN(embeddings, hidden_size=300, dropout_prob=0.3, use_glove=False)
        # Unpack ALL 4 values
        rand_train_acc, rand_dev_acc, rand_train_loss, rand_dev_loss = run_experiment(
            model_rand, train_loader, test_loader, 
            lr=0.0001,          
            weight_decay=1e-5, 
            patience=15      
        )


        # GloVe Detailed (Train vs Dev)
        plt.figure(figsize=(8, 6))
        plt.plot(glove_train_acc, label='GloVe Train Acc', linestyle='--')
        plt.plot(glove_dev_acc, label='GloVe Dev Acc', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('GloVe Model 300d: Train vs Dev')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_glove_full.png'))
        print(f"\nSaved: results/dan_glove_full.png")

        # Random Detailed (Train vs Dev)
        plt.figure(figsize=(8, 6))
        plt.plot(rand_train_acc, label='Random Train Acc', linestyle='--')
        plt.plot(rand_dev_acc, label='Random Dev Acc', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Random Init Model 300d: Train vs Dev')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_random_full.png'))
        print(f"Saved: results/dan_random_full.png")

        # Comparison 
        plt.figure(figsize=(10, 7))
        plt.plot(glove_train_acc, 'b--', alpha=0.5, label='GloVe Train')
        plt.plot(glove_dev_acc, 'b', linewidth=2, label='GloVe Dev')
        plt.plot(rand_train_acc, 'r--', alpha=0.5, label='Random Train')
        plt.plot(rand_dev_acc, 'r', linewidth=2, label='Random Dev')
        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Impact of Pre-training: GloVe vs Random (Full View) 300d')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_all_accuracy.png'))
        print(f"Saved: results/dan_all_accuracy.png")

        # Training Loss Comparison (GloVe vs Random)
        plt.figure(figsize=(10, 7))
        plt.plot(glove_train_loss, 'b--', alpha=0.7, label='GloVe Train Loss')
        plt.plot(rand_train_loss, 'r--', alpha=0.7, label='Random Train Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('NLL Loss (Lower is Better)')
        plt.title('Convergence Speed : Training Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_loss_convergence.png'))
        print(f"Saved: results/dan_loss_convergence.png")

        # GloVe Loss Analysis (Train vs Dev)
        plt.figure(figsize=(8, 6))
        plt.plot(glove_train_loss, label='Train Loss', linestyle='--')
        plt.plot(glove_dev_loss, label='Dev Loss', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('NLL Loss')
        plt.title('GloVe Model 300d: Loss Analysis (Check for Overfitting)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_glove_loss.png'))
        print(f"Saved: results/dan_glove_loss.png")

        # Random Loss Analysis (Train vs Dev)
        plt.figure(figsize=(8, 6))
        plt.plot(rand_train_loss, label='Train Loss', linestyle='--')
        plt.plot(rand_dev_loss, label='Dev Loss', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('NLL Loss')
        plt.title('Random Init Model 300d: Loss Analysis')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(results_dir, 'dan_random_loss.png'))
        print(f"Saved: results/dan_random_loss.png")

    elif args.model == "SUBWORDDAN":
        print("--- Starting BPE Experiments ---")
        vocab_sizes = [1000,2000,5000]
        
    
        fig_acc = plt.figure(figsize=(10, 7))
        ax_acc = fig_acc.add_subplot(111)
        
        fig_loss = plt.figure(figsize=(10, 7))
        ax_loss = fig_loss.add_subplot(111)
        
        for v_size in vocab_sizes:
            print(f"\nTraining BPE Tokenizer (Size {v_size})...")
            tokenizer = BPETokenizer()
            tokenizer.train("data/train.txt", vocab_size=v_size)
            
            train_data = SentimentDatasetBPE("data/train.txt", tokenizer)
            dev_data = SentimentDatasetBPE("data/dev.txt", tokenizer)
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_fn_pad)
            test_loader = DataLoader(dev_data, batch_size=64, shuffle=False, collate_fn=collate_fn_pad)
            
            # BPE
            print(f"Training Model (Vocab {v_size})...")
            model = SubwordDAN(
                hidden_size=300,      
                dropout_prob=0.5,     
                vocab_size=v_size+10, 
                embed_dim=300         
            )
            
            train_acc, dev_acc, train_loss, dev_loss = run_experiment(
                model, train_loader, test_loader,
                lr=0.001,           
                weight_decay=1e-4,
                patience=20
            )
            

            ax_acc.plot(dev_acc, label=f'Vocab {v_size} (Peak: {max(dev_acc):.3f})')
            
            ax_loss.plot(train_loss, label=f'Vocab {v_size} Loss')
        
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        ax_acc.set_title('BPE: Impact of Vocabulary Size (Accuracy)')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Dev Accuracy')
        ax_acc.legend()
        ax_acc.grid()
        fig_acc.savefig(os.path.join(results_dir, 'bpe_results.png'))
        print("Saved bpe_results.png")

        # Save Loss
        ax_loss.set_title('BPE: Convergence Speed (Training Loss)')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('NLL Loss')
        ax_loss.legend()
        ax_loss.grid()
        fig_loss.savefig(os.path.join(results_dir, 'bpe_loss.png'))
        print("Saved bpe_loss.png")

if __name__ == "__main__":
    main()
