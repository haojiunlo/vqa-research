import math
import re

import lightning.pytorch as pl
import numpy as np
import torch
from nltk import edit_distance
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

from src.models.vqa_model import VQAModel

# from transformers import AutoTokenizer


class LitVqaModel(pl.LightningModule):
    def __init__(self, args, dec_tokenizer):
        super().__init__()
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters(args)

        # Now possible to access hyperparameters from hparams
        self.model = VQAModel(
            pretrained_img_enc=self.hparams.pretrained_img_enc,
            pretrained_dec=self.hparams.pretrained_dec,
            pretrained_ocr_enc=self.hparams.pretrained_ocr_enc,
            dec_tokenizer=dec_tokenizer,
        )
        # self.ocr_tokenizer = AutoTokenizer.from_pretrained(
        #     self.hparams.pretrained_ocr_enc
        # )

        self.validation_step_outputs = []

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        # print(f"ocr text: {self.ocr_tokenizer.batch_decode(batch['ocr_text_tensor'])}".replace("[PAD]", ""))
        # print(f"input_ids: {self.model.decoder_tokenizer.batch_decode(batch['input_ids'])}")
        # labels = batch["labels"]
        # labels[labels == -100] = self.model.decoder_tokenizer.pad_token_id
        # print(f"label: {self.model.decoder_tokenizer.batch_decode(labels)}")
        loss = self.model(
            image_tensors=batch["image"],
            decoder_input_ids=batch["input_ids"],
            ocr_text_tensors=batch["ocr_text_tensor"],
            ocr_text_attention_mask=batch["ocr_text_attention_mask"],
            bbox=batch["ocr_bbox"],
            decoder_labels=batch["labels"],
        ).loss
        self.log_dict({"train_loss": loss}, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model.inference(
            image_tensors=batch["image"],
            decoder_input_ids=batch["input_ids"],
            ocr_text_tensors=batch["ocr_text_tensor"],
            ocr_text_attention_mask=batch["ocr_text_attention_mask"],
            bbox_tensor=batch["ocr_bbox"],
            decoder_tokenizer=self.model.decoder_tokenizer,
        )["predictions"]
        print(preds)

        labels = batch["labels"]
        labels[labels == -100] = self.model.decoder_tokenizer.pad_token_id
        answers = self.model.decoder_tokenizer.batch_decode(labels)

        scores = []
        for pred, answer in zip(preds, answers):
            pred = re.findall(
                r"{}(.*?){}".format(
                    "<s_a>",
                    self.model.decoder_tokenizer.eos_token,
                ),
                pred,
            )
            pred = pred[0] if len(pred) > 0 else ""
            answer = answer.replace(self.model.decoder_tokenizer.eos_token, "").replace(
                self.model.decoder_tokenizer.pad_token, ""
            )
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if len(scores) == 1:
                self.print(f"Prediction: {pred}")
                self.print(f"    Answer: {answer}")
                self.print(f" Normed ED: {scores[0]}")

        self.validation_step_outputs.extend(scores)

    def on_validation_epoch_end(self):
        score = np.mean(self.validation_step_outputs)

        self.log_dict({"val_mean_Normed_ED": score}, sync_dist=True, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        max_iter = None

        if self.hparams.max_epochs > 0:
            max_iter = (
                self.hparams.max_epochs * self.hparams.num_training_samples_per_epoch
            ) / (
                self.hparams.train_batch_sizes * torch.cuda.device_count()
                if self.hparams.accelerator == "gpu"
                else 1
            )

        if self.hparams.max_steps > 0:
            max_iter = (
                min(self.hparams.max_steps, max_iter)
                if max_iter is not None
                else self.hparams.max_steps
            )

        # assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": self.cosine_scheduler(
                optimizer, max_iter, self.hparams.warmup_steps
            ),
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
