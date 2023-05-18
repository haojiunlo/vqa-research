import json
import os
from collections import Counter, namedtuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src.models.hash_embedding import HashEmbedding

TextVqaSample = namedtuple(
    "TextVqaSample",
    ["question", "answer", "image_id", "img_width", "img_height", "ocr_info"],
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


class TextVqaDataset(Dataset):
    def __init__(
        self,
        path: str = "data/TextVQA",
        pretrained_vit: str = "google/vit-base-patch16-224",
        pretrained_dec: str = "sshleifer/tiny-mbart",
        hash_embed_n_tok: int = 6000,  # Number of rows in the embedding
        hash_embed_n_hash: int = 4,  # Number of hash functions
    ):
        self.base_path = path
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_vit)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dec)
        self.hash_embed_n_tok = hash_embed_n_tok
        self.hash_embed_n_hash = hash_embed_n_hash

        with open(os.path.join(path, "TextVQA_0.5.1_train.json"), "r") as f:
            data = json.load(f)

        with open(os.path.join(path, "TextVQA_Rosetta_OCR_v0.2_train.json"), "r") as f:
            ocr_data = json.load(f)

        img_data = [
            {
                "question": x["question"],
                "answer": Counter(x["answers"]).most_common(1)[0][0],
                "image_id": x["image_id"],
                "img_width": x["image_width"],
                "img_height": x["image_height"],
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
                v["img_width"],
                v["img_height"],
                [
                    # Note: `+ 1` for x, y img coords to account for embedding pad idx
                    OcrInfo(
                        x["word"],
                        int(v["img_width"] * x["w"]),
                        int(v["img_height"] * x["h"]),
                        int(v["img_width"] * x["x0"]) + 1,
                        int(v["img_height"] * x["y0"]) + 1,
                        int(v["img_width"] * x["x0"])
                        + int(v["img_width"] * x["w"])
                        + 1,
                        int(v["img_height"] * x["y0"])
                        + int(v["img_height"] * x["h"])
                        + 1,
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
        question_tensor = self.decoder_tokenizer(
            vqa_sample.question,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=128,  # TODO -- don't hardcode this
            padding="max_length",
        ).input_ids.squeeze(0)

        answer_tensor = self.decoder_tokenizer(
            vqa_sample.answer,
            add_special_tokens=False,
            return_tensors="pt",
            max_length=128,  # TODO -- don't hardcode this
            padding="max_length",
        ).input_ids.squeeze(0)

        # Preprocess the image
        im = Image.open(
            os.path.join(self.base_path, "train_images", f"{vqa_sample.image_id}.jpg")
        )
        im = im.convert("RGB")
        image_tensor = self.image_processor(
            im, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Prepare OCR Embedding input
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

        return {
            "question": question_tensor,
            "answer": answer_tensor,
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
    ds = TextVqaDataset(
        path="/Users/charles/Projects/brew/brew-1146-vqa-research/data/TextVQA"
    )
    sample = ds[100]
    print(sample)

    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate_batch)
    for batch in dl:
        print(batch)
