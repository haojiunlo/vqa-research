import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from src.models.lightning_module import LitVqaModel
from src.models.ocr_embedding import OcrEmbedding
from src.utils.datasets import CustomDataCollator, TextVqaDataset


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_img_enc",
        default="google/vit-base-patch16-224",
        type=str,
        help="pretrained_img_enc",
    )
    parser.add_argument(
        "--pretrained_dec",
        default="sshleifer/student-bart-base-3-3",
        type=str,
        help="pretrained_dec",
    )
    parser.add_argument(
        "--pretrained_ocr_enc",
        default="microsoft/layoutlm-base-uncased",
        type=str,
        help="pretrained_ocr_enc",
    )
    parser.add_argument("--accelerator", default="gpu", type=str, help="accelerator")
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Accumulates gradients over k batches before stepping the optimizer.",
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
    parser.add_argument("--exp_name", default="default", type=str, help="exp_name")
    parser.add_argument(
        "--exp_version", default="default_ver", type=str, help="exp_version"
    )
    parser.add_argument("--max_epochs", default=30, type=int, help="max epochs")
    parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
    parser.add_argument("--precision", default=16, type=int, help="prcision")
    parser.add_argument(
        "--gradient_clip_val", default=1.0, type=float, help="Gradient clipping value"
    )
    parser.add_argument(
        "--train_batch_sizes", default=8, type=int, help="train_batch_sizes"
    )
    parser.add_argument(
        "--val_batch_sizes", default=1, type=int, help="val_batch_sizes"
    )
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup_steps")
    parser.add_argument("--fast_dev_run", default=False, type=bool, help="fast_dev_run")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    trn_dataset = TextVqaDataset(
        path="/home/jovyan/vol-1/BREW-1146/data/TextVQA",
        pretrained_vit=args.pretrained_img_enc,
        pretrained_dec=args.pretrained_dec,
        pretrained_ocr_enc=args.pretrained_ocr_enc,
        mode="train",
    )
    val_dataset = TextVqaDataset(
        path="/home/jovyan/vol-1/BREW-1146/data/TextVQA",
        pretrained_vit=args.pretrained_img_enc,
        pretrained_dec=args.pretrained_dec,
        pretrained_ocr_enc=args.pretrained_ocr_enc,
        mode="val",
    )

    train_loader = DataLoader(
        trn_dataset,
        batch_size=args.train_batch_sizes,
        shuffle=True,
        collate_fn=CustomDataCollator(trn_dataset.decoder_tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_sizes,
        collate_fn=CustomDataCollator(val_dataset.decoder_tokenizer),
    )

    num_training_samples_per_epoch = len(trn_dataset)

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.result_path) / args.exp_name / args.exp_version,
    )
    callbacks = [lr_callback, checkpoint_callback]

    logger = (
        MLFlowLogger(
            run_name=args.exp_version,
            experiment_name=args.exp_name,
            tracking_uri=args.mlflow_tracking_uri,
            save_dir=args.result_path,
            log_model=True,
        )
        if args.mlflow_tracking_uri is not None
        else TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    )

    model = LitVqaModel(
        {**vars(args), "num_training_samples_per_epoch": num_training_samples_per_epoch}
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        fast_dev_run=args.fast_dev_run,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
