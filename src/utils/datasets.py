import json
import os
from collections import Counter, namedtuple
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src.models.hash_embedding import HashEmbedding

TextVqaSample = namedtuple(
    "TextVqaSample",
    ["question", "answer", "image_id", "ocr_info"],
)

OcrInfo = namedtuple("OcrInfo", ["word", "w", "h", "x0", "y0", "x1", "y1"])


def collate_batch(samples):
    batch = {
        "question": torch.stack([x["question"] for x in samples]),
        "answer": torch.stack([x["answer"] for x in samples]),
        "image": torch.stack([x["image"] for x in samples]),
        # Note: padding value is 0, the same as the padding index in the embeddings
        "tok": pad_sequence([x["ocr"]["tok"] for x in samples], batch_first=True),
        "x0": pad_sequence([x["ocr"]["x0"] for x in samples], batch_first=True),
        "y0": pad_sequence([x["ocr"]["y0"] for x in samples], batch_first=True),
        "x1": pad_sequence([x["ocr"]["x1"] for x in samples], batch_first=True),
        "y1": pad_sequence([x["ocr"]["y1"] for x in samples], batch_first=True),
        "w": pad_sequence([x["ocr"]["w"] for x in samples], batch_first=True),
        "h": pad_sequence([x["ocr"]["h"] for x in samples], batch_first=True),
    }

    return batch


@dataclass
class CustomDataCollator:
    tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, samples: dict):
        labels = (
            [feature["labels"] for feature in samples]
            if "labels" in samples[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            padding_side = self.tokenizer.padding_side
            for sample in samples:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(sample["labels"])
                )
                if isinstance(sample["labels"], list):
                    sample["labels"] = (
                        sample["labels"] + remainder
                        if padding_side == "right"
                        else remainder + sample["labels"]
                    )
                elif padding_side == "right":
                    sample["labels"] = np.concatenate(
                        [sample["labels"], remainder]
                    ).astype(np.int64)
                else:
                    sample["labels"] = np.concatenate(
                        [remainder, sample["labels"]]
                    ).astype(np.int64)
        encoding = [
            {"input_ids": x["input_ids"], "labels": x["labels"]} for x in samples
        ]
        batch = self.tokenizer.pad(encoding, return_tensors="pt")
        batch.update(
            {
                "image": torch.stack([x["image"] for x in samples]),
                "ocr_text_tensor": torch.stack([x["ocr_text_tensor"] for x in samples]),
                "ocr_text_attention_mask": torch.stack(
                    [x["ocr_text_attention_mask"] for x in samples]
                ),
                "ocr_bbox": torch.stack([x["ocr_bbox"] for x in samples]),
            }
        )
        return batch


class TextVqaDataset(Dataset):
    def __init__(
        self,
        path: str = "data/TextVQA",
        pretrained_vit: str = "google/vit-base-patch16-224",
        pretrained_dec: str = "sshleifer/tiny-mbart",
        pretrained_ocr_enc: str = "microsoft/layoutlm-base-uncased",
        hash_embed_n_tok: int = 6000,  # Number of rows in the embedding
        hash_embed_n_hash: int = 4,  # Number of hash functions
    ):
        self.base_path = path
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_vit)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dec)

        self.pretrained_ocr_enc = pretrained_ocr_enc
        if pretrained_ocr_enc is not None:
            self.orc_text_tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_ocr_enc
            )
        else:
            self.hash_embed_n_tok = hash_embed_n_tok
            self.hash_embed_n_hash = hash_embed_n_hash

        with open(os.path.join(path, "TextVQA_0.5.1_train.json"), "r") as f:
            data = json.load(f)

        with open(os.path.join(path, "TextVQA_Rosetta_OCR_v0.2_train.json"), "r") as f:
            ocr_data = json.load(f)

        img_data = [
            {
                "question": x["question"],
                "answer": Counter(x.get("answers", ["default answer"])).most_common(1)[
                    0
                ][0],
                "image_id": x["image_id"],
            }
            for x in data["data"]
        ]

        imgid2ocr = {
            x["image_id"]: [
                {
                    "word": y["word"],
                    "w": y["bounding_box"]["width"],
                    "h": y["bounding_box"]["height"],
                    "x0": y["bounding_box"]["top_left_x"],
                    "y0": y["bounding_box"]["top_left_y"],
                }
                for y in x["ocr_info"]
            ]
            for x in ocr_data["data"]
        }

        self.samples = [
            TextVqaSample(
                v["question"],
                v["answer"],
                v["image_id"],
                [
                    # Note: `+ 1` for x, y img coords to account for embedding pad idx
                    # Note: Scale all img coords onto a 1000 x 1000 image
                    OcrInfo(
                        x["word"],
                        int(1000 * x["w"]),
                        int(1000 * x["h"]),
                        int(1000 * x["x0"]) + 1,
                        int(1000 * x["y0"]) + 1,
                        int(1000 * x["x0"]) + int(1000 * x["w"]) + 1,
                        int(1000 * x["y0"]) + int(1000 * x["h"]) + 1,
                    )
                    for x in imgid2ocr.get(v["image_id"])
                ],
            )
            for v in img_data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vqa_sample = self.samples[idx]

        # Tokenize question and answer using the decoders tokenizer
        input_ids = self.decoder_tokenizer(
            vqa_sample.question,
            text_pair=vqa_sample.answer,
            add_special_tokens=True,
            max_length=128,  # TODO -- don't hardcode this
            truncation=True,
        ).input_ids

        # Preprocess the image
        im = Image.open(
            os.path.join(self.base_path, "train_images", f"{vqa_sample.image_id}.jpg")
        )
        im = im.convert("RGB")
        image_tensor = self.image_processor(
            im, return_tensors="pt"
        ).pixel_values.squeeze(0)
        if self.pretrained_ocr_enc:
            ocr_text = [x.word for x in vqa_sample.ocr_info]
            normalized_word_boxes = [
                [x.x0, x.y0, x.x1, x.y1] for x in vqa_sample.ocr_info
            ]
            token_boxes = []
            for word, box in zip(ocr_text, normalized_word_boxes):
                word_tokens = self.orc_text_tokenizer.tokenize(word)
                token_boxes.extend([box] * len(word_tokens))

            # add bounding boxes of cls + sep tokens
            if len(token_boxes) >= 197:
                token_boxes = (
                    [[0, 0, 0, 0]]
                    + token_boxes[:195]
                    # TODO -- don't hardcode this
                    + [[1000, 1000, 1000, 1000]]
                )  # truncate to seq_length
            else:
                token_boxes = (
                    [[0, 0, 0, 0]]
                    + token_boxes
                    # TODO -- don't hardcode this
                    + [[1000, 1000, 1000, 1000]] * (197 - 1 - len(token_boxes))
                )  # pad to seq_length
            bbox = torch.tensor(token_boxes)

            ocr_text_input = self.orc_text_tokenizer(
                " ".join(ocr_text),
                return_tensors="pt",
                truncation=True,
                max_length=197,  # TODO -- don't hardcode this
                padding="max_length",
            )
            ocr_text_input_ids = ocr_text_input["input_ids"].squeeze(0)
            ocr_text_attention_mask = ocr_text_input["attention_mask"].squeeze(0)

            labels = [
                id if i > input_ids.index(self.decoder_tokenizer.eos_token_id) else -100
                for i, id in enumerate(input_ids)
            ]
            # model doesn't need to predict prompt (for VQA)

            return {
                "input_ids": input_ids,
                "labels": labels,
                "image": image_tensor,
                "ocr_text_tensor": ocr_text_input_ids,
                "ocr_text_attention_mask": ocr_text_attention_mask,
                "ocr_bbox": bbox,
            }
        else:
            # Prepare OCR Embedding input
            # Note: not all images have OCR data
            if vqa_sample.ocr_info:
                tok_tensor = HashEmbedding.dataset_prepare_input(
                    [x.word for x in vqa_sample.ocr_info],
                    n_tok=self.hash_embed_n_tok,
                    n_hash=self.hash_embed_n_hash,
                )
                x0_tensor = torch.LongTensor([x.x0 for x in vqa_sample.ocr_info])
                y0_tensor = torch.LongTensor([x.y0 for x in vqa_sample.ocr_info])
                x1_tensor = torch.LongTensor([x.x1 for x in vqa_sample.ocr_info])
                y1_tensor = torch.LongTensor([x.y1 for x in vqa_sample.ocr_info])
                w_tensor = torch.LongTensor([x.w for x in vqa_sample.ocr_info])
                h_tensor = torch.LongTensor([x.h for x in vqa_sample.ocr_info])
            else:
                tok_tensor = torch.LongTensor([[0, 0, 0, 0]])
                x0_tensor = torch.LongTensor([0])
                y0_tensor = torch.LongTensor([0])
                x1_tensor = torch.LongTensor([0])
                y1_tensor = torch.LongTensor([0])
                w_tensor = torch.LongTensor([0])
                h_tensor = torch.LongTensor([0])

        return {
            "input_tensor": input_tensor,
            "answer": labels,
            "image": image_tensor,
            "ocr": {
                "tok": tok_tensor,
                "x0": x0_tensor,
                "y0": y0_tensor,
                "x1": x1_tensor,
                "y1": y1_tensor,
                "w": w_tensor,
                "h": h_tensor,
            },
        }


if __name__ == "__main__":
    ds = TextVqaDataset(path="dataset")
    sample = ds[100]
    print(sample)

    dl = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=CustomDataCollator(ds.decoder_tokenizer),
    )
    for d in dl:
        print(d)
        break
