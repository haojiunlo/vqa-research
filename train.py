import lightning.pytorch as pl
from torch.utils.data import DataLoader

from models.ocr_embedding import OcrEmbedding
from models.vqa_model import LitVqaModel
from utils.datasets import VqaDataset

if __name__ == "__main__":
    dataset = VqaDataset()
    train_loader = DataLoader(dataset)

    model = LitVqaModel(OcrEmbedding, None, None)

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)
