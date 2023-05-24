import json
import os
from collections import Counter
from typing import Union

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer, PreTrainedTokenizer

from src.datasets.collators import CustomDataCollator, collate_batch
from src.datasets.common import OcrInfo, VqaSample, vqa_sample_2_tensor


class TextVqaDataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str = "train",
        pretrained_vit: str = "google/vit-base-patch16-224",
        pretrained_dec: Union[
            PreTrainedTokenizer, str
        ] = "sshleifer/student-bart-base-3-3",
        max_len: int = 128,
        pretrained_ocr_enc: str = "microsoft/layoutlm-base-uncased",
        hash_embed_n_tok: int = 6000,  # Number of rows in the embedding
        hash_embed_n_hash: int = 4,  # Number of hash functions
    ):
        self.base_path = path
        self.mode = mode
        if self.mode == "train":
            self.input_folder = "TextVQA_0.5.1_train.json"
            self.ocr_folder = "TextVQA_Rosetta_OCR_v0.2_train.json"
            self.image_folder = "train_images"
        elif self.mode == "val":
            self.input_folder = "TextVQA_0.5.1_val.json"
            self.ocr_folder = "TextVQA_Rosetta_OCR_v0.2_val.json"
            self.image_folder = "train_images"
        else:
            raise NotImplementedError("model should be one of `train` or `val`")

        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_vit)

        if isinstance(pretrained_dec, str):
            self.decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dec)
        else:
            self.decoder_tokenizer = pretrained_dec
        self.decoder_tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    "<s_ocr>",
                    "</s_ocr>",
                    "<s_q>",
                    "</s_q>",
                    "<s_a>",
                ]
            }
        )
        self.max_length = max_len

        self.pretrained_ocr_enc = pretrained_ocr_enc
        if pretrained_ocr_enc is not None:
            self.ocr_text_tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_ocr_enc
            )
        else:
            self.ocr_text_tokenizer = None

        self.hash_embed_n_tok = hash_embed_n_tok
        self.hash_embed_n_hash = hash_embed_n_hash

        with open(os.path.join(path, self.input_folder), "r") as f:
            data = json.load(f)

        with open(os.path.join(path, self.ocr_folder), "r") as f:
            ocr_data = json.load(f)

        img_data = [
            {
                "question": x["question"],
                "answer": Counter(x.get("answers", ["no answer provided"])).most_common(
                    1
                )[0][0],
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
            VqaSample(
                v["question"],
                v["answer"],
                v["image_id"],
                [
                    # Note: `+ 1` for x, y img coords to account for embedding pad idx
                    # Note: Scale all img coords onto a 1000 x 1000 image
                    # FIXME -- make sure these are between 0 and 1000
                    OcrInfo(
                        x["word"],
                        int(1000 * x["w"]),
                        int(1000 * x["h"]),
                        max(0, min(1000, int(1000 * x["x0"]))),
                        max(0, min(1000, int(1000 * x["y0"]))),
                        max(0, min(1000, int(1000 * x["x0"]) + int(1000 * x["w"]))),
                        max(0, min(1000, int(1000 * x["y0"]) + int(1000 * x["h"]))),
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

        return vqa_sample_2_tensor(
            vqa_sample,
            self.decoder_tokenizer,
            self.max_length,
            os.path.join(self.base_path, "train_images", f"{vqa_sample.image_id}.jpg"),
            self.image_processor,
            self.ocr_text_tokenizer,
            self.hash_embed_n_tok,
            self.hash_embed_n_hash,
            self.mode,
        )


if __name__ == "__main__":
    ds = TextVqaDataset(path="../../data/TextVQA", pretrained_ocr_enc=None)
    sample = ds[100]
    print(sample)

    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=CustomDataCollator())
    for batch in dl:
        print(batch)
