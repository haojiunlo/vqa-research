import argparse

import lightning.pytorch as pl
import torch

from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.mlflow import MLFlowLogger

from src.models.lightning_module import LitVqaModel
from src.models.ocr_embedding import OcrEmbedding
from src.utils.datasets import CustomDataCollator, TextVqaDataset


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accelerator", default="gpu", type=str, help="accelerator"
    )
    parser.add_argument(
        "--accumulate_grad_batches", default=1, type=int, help="Accumulates gradients over k batches before stepping the optimizer."
    )
    parser.add_argument(
        "--learning_rate", default=3e-5, type=float, help="learning rate"
    )
    parser.add_argument(
        "--mlflow_tracking_uri", default=None, type=str, help="mlflow_tracking_uri"
    )
    parser.add_argument(
        "--result_path", default="results", type=str, help="result_path"
    )
    parser.add_argument(
        "--exp_name", default="default", type=str, help="exp_name"
    )
    parser.add_argument(
        "--exp_version", default="default_ver", type=str, help="exp_version"
    )
    parser.add_argument(
        "--max_epochs", default=30, type=int, help="max epochs"
    )
    parser.add_argument(
        "--max_steps", default=-1, type=int, help="max steps"
    )
    parser.add_argument(
        "--precision", default=16, type=int, help="prcision"
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.0, type=float, help="Gradient clipping value"
    )
    parser.add_argument(
        "--train_batch_sizes", default=8, type=int, help="train_batch_sizes"
    )
    parser.add_argument(
        "--val_batch_sizes", default=1, type=int, help="val_batch_sizes"
    )
    parser.add_argument(
        "--warmup_steps", default=500, type=int, help="warmup_steps"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    dataset = TextVqaDataset(path="dataset")
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_sizes,
        shuffle=True,
        collate_fn=CustomDataCollator(dataset.decoder_tokenizer),
    )

    num_training_samples_per_epoch = len(dataset)

    logger = MLFlowLogger(
        experiment_name = args.exp_name,
        tracking_uri = args.mlflow_tracking_uri,
        save_dir = args.result_path
    )

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(args.result_path) / args.exp_name / args.exp_version,
        filename="artifacts",
        save_top_k=1,
        save_last=False,
        mode="min",
    )

    model = LitVqaModel({**vars(args), "num_training_samples_per_epoch": num_training_samples_per_epoch})

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val
    )
    trainer.fit(model=model, train_dataloaders=train_loader)
