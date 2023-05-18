import json
import os
from collections import Counter

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src.datasets.common import OcrInfo, VqaSample, collate_batch, vqa_sample_2_tensor


class TextVqaDataset(Dataset):
    def __init__(
        self,
        path: str,
        pretrained_vit: str = "google/vit-base-patch16-224",
        pretrained_dec: str = "sshleifer/tiny-mbart",
        max_len: int = 128,
        hash_embed_n_tok: int = 6000,  # Number of rows in the embedding
        hash_embed_n_hash: int = 4,  # Number of hash functions
    ):
        self.base_path = path
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_vit)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dec)
        self.max_length = max_len
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
                    OcrInfo(
                        x["word"],
                        int(1000 * x["w"]) + 1,
                        int(1000 * x["h"]) + 1,
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

        return vqa_sample_2_tensor(
            vqa_sample,
            self.decoder_tokenizer,
            self.max_length,
            os.path.join(self.base_path, "train_images", f"{vqa_sample.image_id}.jpg"),
            self.image_processor,
            self.hash_embed_n_tok,
            self.hash_embed_n_hash,
        )


if __name__ == "__main__":
    ds = TextVqaDataset(path="../../data/TextVQA")
    sample = ds[100]
    print(sample)

    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate_batch)
    for batch in dl:
        print(batch)
