import torch
from torch import nn


class OcrEmbedding(nn.Module):
    def __init__(
        self, n_tok: int, emb_dim: int, img_width: int, img_height: int, alpha: float
    ):
        super().__init__()
        # OCR Token Embedding
        self.E_ocr = nn.Embedding(n_tok, emb_dim)
        # OCR Bounding Box Embedding
        self.alpha = alpha
        self.E_x = nn.Embedding(img_width, emb_dim)
        self.E_y = nn.Embedding(img_height, emb_dim)
        self.E_w = nn.Embedding(img_width, emb_dim)
        self.E_h = nn.Embedding(img_height, emb_dim)

        self.MLP = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())

    def forward(self, tok, x0, y0, x1, y1, w, h):
        tok_emb = self.E_ocr(tok)
        bbox_emb = self.MLP(
            self.E_x(x0)
            + self.E_y(y0)
            + self.E_x(x1)
            + self.E_y(y1)
            + self.E_w(w)
            + self.E_h(h)
        )
        y = tok_emb + self.alpha * bbox_emb
        return y


if __name__ == "__main__":
    ocr_emb = OcrEmbedding(10, 64, 100, 100, 0.5)

    test = ocr_emb(
        # Word Token
        torch.LongTensor([5]),
        # Bounding Box
        torch.LongTensor([5]),
        torch.LongTensor([5]),
        torch.LongTensor([5]),
        torch.LongTensor([5]),
        torch.LongTensor([5]),
        torch.LongTensor([5]),
    )
    print(test.shape)
