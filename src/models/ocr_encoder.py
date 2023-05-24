import torch
from torch import nn

from src.models.ocr_embedding import OcrEmbedding


class OcrEncoder(nn.Module):
    def __init__(
        self,
        # Embed params
        n_tok: int = 6000,
        n_img_emb: int = 1000,
        emb_dim: int = 512,
        alpha: float = 0.5,
        # Encoder params
        d_model: int = 512,
        n_head: int = 8,
        n_layers: int = 6,
    ):
        super().__init__()
        self.embed = OcrEmbedding(
            n_tok=n_tok,
            n_img_emb=n_img_emb,
            emb_dim=emb_dim,
            alpha=alpha,
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.encode = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def prepare_input(self, input):
        # TODO
        pass

    def forward(self, batch):
        emb = self.embed(
            batch["tok"],
            batch["x0"],
            batch["y0"],
            batch["x1"],
            batch["y1"],
            batch["w"],
            batch["h"],
        )
        enc = self.encode(emb)
        return enc


if __name__ == "__main__":
    encoder = OcrEncoder()
    test = encoder(
        {
            "tok": torch.LongTensor([[[5, 5, 5, 5], [4, 4, 4, 4], [3, 3, 3, 3]]]),
            "x0": torch.LongTensor([[5, 4, 3]]),
            "y0": torch.LongTensor([[5, 4, 3]]),
            "x1": torch.LongTensor([[5, 4, 3]]),
            "y1": torch.LongTensor([[5, 4, 3]]),
            "w": torch.LongTensor([[5, 4, 3]]),
            "h": torch.LongTensor([[5, 4, 3]]),
        }
    )
    print(test.shape)
