from typing import List

import mmh3
import torch
from torch import nn


class HashEmbedding(nn.Module):
    def __init__(self, n_tok: int, emb_dim: int, n_hash: int):
        super().__init__()
        self.n_tok = n_tok
        self.n_hash = n_hash
        self.E = nn.Embedding(n_tok + 1, emb_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.E.weight)

    def prepare_input(self, tokens: List[str]):
        all_keys = [
            torch.LongTensor(
                [(mmh3.hash(t, i) % self.n_tok) for i in range(self.n_hash)]
            )
            for t in tokens
        ]
        # [n_tokens x n_keys]
        return torch.stack(all_keys)

    @staticmethod
    def dataset_prepare_input(tokens: List[str], n_tok: int, n_hash: int):
        # Note: `(mmh3.hash(t, i) % n_tok) + 1)` - the `+ 1` is to account for padding
        all_keys = [
            torch.LongTensor([((mmh3.hash(t, i) % n_tok) + 1) for i in range(n_hash)])
            for t in tokens
        ]
        # [n_tokens x n_keys]
        return torch.stack(all_keys)

    def forward(self, x):
        # X: [batch x n_tokens x n_keys]
        embeds = self.E(x)
        # [batch x n_tokens x n_keys x emb_dim]
        y = embeds.sum(dim=2)
        # [batch x n_tokens x emb_dim]
        return y


if __name__ == "__main__":
    hash_embed = HashEmbedding(100, 32, 4)

    lookup = hash_embed.prepare_input(["This", "is", "a", "test", "sentence"])
    embeds = hash_embed(lookup.unsqueeze(0))
    print(embeds.shape)
