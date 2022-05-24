import os
import torch
import torch.nn as nn
from Source.utils import save_file


class SkipGram(nn.Module):

    def __init__(self, vocab_len, device, embedding_size=64):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_len, embedding_size).to(device)
        self.weights = torch.empty(embedding_size, vocab_len, requires_grad=True).type(torch.FloatTensor).to(device)
        _ = torch.nn.init.normal_(self.weights)
        self.out = nn.LogSigmoid()
        self.device = device

    def forward(self, center_word, context_words):
        embeddings_ = self.embeddings(center_word)
        weights_ = self.weights[:, context_words]
        output = torch.einsum('bi,ibo->bo', embeddings_, weights_)
        true_y = torch.zeros(output.shape[0], dtype=torch.int64).to(self.device)
        return self.out(output), true_y

    def save_files(self, out_path="Output"):
        save_file(os.path.join(out_path, "emb.pkl"), self.embeddings)
        save_file(os.path.join(out_path, "weights.pkl"), self.weights)
