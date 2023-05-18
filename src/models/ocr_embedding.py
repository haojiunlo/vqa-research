import torch
from torch import nn

from src.models.hash_embedding import HashEmbedding


class OcrEmbedding(nn.Module):
    def __init__(self, n_tok: int, n_img_emb: int, emb_dim: int, alpha: float):
        super().__init__()
        # OCR Token Embedding
        # Note: HashEmbedding inits its own weights and accounts for padding internally
        self.E_ocr = HashEmbedding(n_tok, emb_dim, 4)

        # OCR Bounding Box Embedding
        self.alpha = alpha
        self.E_x = nn.Embedding(n_img_emb + 1, emb_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.E_x.weight)

        self.E_y = nn.Embedding(n_img_emb + 1, emb_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.E_y.weight)

        self.E_w = nn.Embedding(n_img_emb + 1, emb_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.E_w.weight)

        self.E_h = nn.Embedding(n_img_emb + 1, emb_dim, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.E_h.weight)

        self.MLP = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.MLP.apply(_init_weights)

    def forward(self, tok, x0, y0, x1, y1, w, h):
        # [batch x n_feats x n_hash]
        tok_emb = self.E_ocr(tok)
        # [batch x n_feats]
        bbox_emb = self.MLP(
            self.E_x(x0)
            + self.E_y(y0)
            + self.E_x(x1)
            + self.E_y(y1)
            + self.E_w(w)
            + self.E_h(h)
        )
        # [batch x n_feats x emb_dim]
        y = tok_emb + self.alpha * bbox_emb
        return y


if __name__ == "__main__":
    ocr_emb = OcrEmbedding(100, 100, 64, 0.5)

    test = ocr_emb(
        # Word Token
        torch.LongTensor([[[5, 5, 5, 5]]]),
        # Bounding Box
        torch.LongTensor([[5]]),
        torch.LongTensor([[5]]),
        torch.LongTensor([[5]]),
        torch.LongTensor([[5]]),
        torch.LongTensor([[5]]),
        torch.LongTensor([[5]]),
    )
    print(test.shape)
