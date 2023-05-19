import math
import re

import lightning.pytorch as pl
import numpy as np
import torch
from nltk import edit_distance
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

from src.models.vqa_model import VQAModel


class LitVqaModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = VQAModel()
        self.lr = args["learning_rate"]
        self.train_batch_sizes = args["train_batch_sizes"]
        self.max_epochs = args["max_epochs"]
        self.max_steps = args["max_steps"]
        self.warmup_steps = args["warmup_steps"]
        self.num_training_samples_per_epoch = args["num_training_samples_per_epoch"]
        self.accelerator = args["accelerator"]

    def training_step(self, batch, batch_idx):
        loss = self.model(
            image_tensors=batch["image"],
            decoder_input_ids=batch["input_ids"],
            ocr_text_tensors=batch["ocr_text_tensor"],
            ocr_text_attention_mask=batch["ocr_text_attention_mask"],
            bbox=batch["ocr_bbox"],
            decoder_labels=batch["labels"],
        )[0]
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    # def validation_step(self, batch, batch_idx, dataset_idx=0):
    #     preds = self.model.inference(
    #         image_tensors=batch["image"],
    #         decoder_input_ids=batch["input_ids"],
    #         ocr_text_tensors=batch["ocr_text_tensor"],
    #         ocr_text_attention_mask=batch["ocr_text_attention_mask"],
    #         bbox_tensor=batch["ocr_bbox"],
    #         decoder_tokenizer=self.model.decoder_tokenizer
    #     )["predictions"]

    #     labels = batch["labels"]
    #     labels = labels[labels == -100] = self.model.decoder_tokenizer.pad_token_id
    #     answers = self.model.decoder_tokenizer.batch_decode(labels)

    #     scores = list()
    #     for pred, answer in zip(preds, answers):
    #         pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    #         answer = re.sub(r"<.*?>", "", answer, count=1)
    #         answer = answer.replace(self.model.decoder.tokenizer.eos_token, "")
    #         scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

    #         if self.config.get("verbose", False) and len(scores) == 1:
    #             self.print(f"Prediction: {pred}")
    #             self.print(f"    Answer: {answer}")
    #             self.print(f" Normed ED: {scores[0]}")

    #     return scores

    # def on_validation_epoch_end(self, validation_step_outputs):
    #     num_of_loaders = len(self.config.dataset_name_or_paths)
    #     if num_of_loaders == 1:
    #         validation_step_outputs = [validation_step_outputs]
    #     assert len(validation_step_outputs) == num_of_loaders
    #     cnt = [0] * num_of_loaders
    #     total_metric = [0] * num_of_loaders
    #     val_metric = [0] * num_of_loaders
    #     for i, results in enumerate(validation_step_outputs):
    #         for scores in results:
    #             cnt[i] += len(scores)
    #             total_metric[i] += np.sum(scores)
    #         val_metric[i] = total_metric[i] / cnt[i]
    #         val_metric_name = f"val_metric_{i}th_dataset"
    #         self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
    #     self.log_dict(
    #         {"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True
    #     )

    def configure_optimizers(self):
        max_iter = None

        if self.max_epochs > 0:
            max_iter = (self.max_epochs * self.num_training_samples_per_epoch) / (
                self.train_batch_sizes * torch.cuda.device_count()
                if self.accelerator == "gpu"
                else 1
            )

        if self.max_steps > 0:
            max_iter = (
                min(self.max_steps, max_iter)
                if max_iter is not None
                else self.max_steps
            )

        # assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, self.warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)
