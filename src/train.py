import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.models.ocr_embedding import OcrEmbedding
from src.models.vqa_model import LitVqaModel
from src.utils.datasets import TextVqaDataset

if __name__ == "__main__":
    dataset = TextVqaDataset()
    train_loader = DataLoader(dataset)

    model = LitVqaModel(OcrEmbedding, None, None)

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)
