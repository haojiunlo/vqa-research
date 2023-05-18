import argparse

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from src.models.lightning_module import LitVqaModel
from src.models.ocr_embedding import OcrEmbedding
from src.utils.datasets import CustomDataCollator, TextVqaDataset


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate", default=3e-5, type=float, help="learning rate"
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    dataset = TextVqaDataset(path="dataset")
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=CustomDataCollator(dataset.decoder_tokenizer),
    )

    model = LitVqaModel(args)

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)
