import os
import torch
import config
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Source.model import SkipGram
from Source.utils import load_file
from Source.data import SkipGramDataset


def train_sg(dataloader, model, criterion, optimizer, device, num_epochs):
    model.train()
    best_loss = 1e8
    patience = 0
    for i in range(num_epochs):
        epoch_loss = []
        print(f"Epoch {i+1} of {num_epochs}")
        for center_word, context_words in tqdm(dataloader):
            center_word = center_word.to(device)
            context_words = context_words.to(device)
            output, true_y = model(center_word, context_words)
            loss = criterion(output, true_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        print(f"Loss: {epoch_loss}")
        if patience == 10:
            print("Early stopping...")
    model.save_files()


def main(args_):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokens = load_file(os.path.join(args_.output_path, args_.token_file))
    dataset = SkipGramDataset(input_data=tokens, context_window=args_.context_window,
                              out_path=args_.output_path, t=args_.t, k=args_.k)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_.batch_size,
                                             shuffle=True, drop_last=True)
    model = SkipGram(dataset.vocab_count, device, embedding_size=args_.embedding_size)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train_sg(dataloader, model, criterion, optimizer, device, args_.num_epochs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_file", type=str, default=config.token_file,
                        help="File containing word tokens")
    parser.add_argument("--output_path", type=str, default=config.output_folder,
                        help="Output folder name")
    parser.add_argument("--context_window", type=int, default=config.context_window,
                        help="Context window size")
    parser.add_argument("--t", type=float, default=config.t,
                        help="Threshold")
    parser.add_argument("--k", type=int, default=config.k,
                        help="Number of negative samples")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help="Batch size of training")
    parser.add_argument("--embedding_size", type=int, default=config.embedding_size,
                        help="Embedding size of word vectors")
    parser.add_argument("--lr", type=float, default=config.lr,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs,
                        help="Number of epochs")
    args = parser.parse_args()
    main(args)
