import json
import os

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src.datasets.collators import CustomDataCollator, collate_batch
from src.datasets.common import OcrInfo, VqaSample, vqa_sample_2_tensor


class InfoVqaDataset(Dataset):
    def __init__(
        self,
        path: str,
        mode: str = "train",
        pretrained_vit: str = "google/vit-base-patch16-224",
        pretrained_dec: str = "sshleifer/tiny-mbart",
        max_len: int = 128,
        pretrained_ocr_enc: str = "microsoft/layoutlm-base-uncased",
        hash_embed_n_tok: int = 6000,  # Number of rows in the embedding
        hash_embed_n_hash: int = 4,  # Number of hash functions
    ):
        self.base_path = path
        self.mode = mode
        if self.mode == "train":
            self.input_folder = "infographicVQA_train_v1.0.json"
            self.ocr_folder = "infographicVQA_train_v1.0_ocr_outputs"
            self.image_folder = "infographicVQA_train_v1.0_images"
        elif self.mode == "val":
            self.input_folder = "infographicVQA_val_v1.0.json"
            self.ocr_folder = "infographicVQA_val_v1.0_ocr_outputs"
            self.image_folder = "infographicVQA_val_v1.0_images"
        else:
            raise NotImplementedError("model should be one of `train` or `val`")

        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_vit)

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(pretrained_dec)
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

        # Note: Info-VQA has multiple valid answers per image
        img_data = [
            {
                "question": x["question"],
                "answer": a,
                "image_fn": x["image_local_name"],
                "ocr_fn": x["ocr_output_file"],
            }
            for x in data["data"]
            for a in x["answers"]
        ]

        for item in img_data:
            with open(os.path.join(path, self.ocr_folder, item["ocr_fn"])) as f:
                ocr_data = json.load(f)

            item["ocr_data"] = [
                {
                    "tok": x["Text"],
                    "w": x["Geometry"]["BoundingBox"]["Width"],
                    "h": x["Geometry"]["BoundingBox"]["Height"],
                    "x0": x["Geometry"]["BoundingBox"]["Left"],
                    "y0": x["Geometry"]["BoundingBox"]["Top"],
                }
                for x in ocr_data.get("WORD", [])
            ]

        self.samples = [
            VqaSample(
                x["question"],
                x["answer"],
                x["image_fn"],
                [
                    OcrInfo(
                        y["tok"],
                        int(1000 * y["w"]) + 1,
                        int(1000 * y["h"]) + 1,
                        max(0, min(1000, int(1000 * y["x0"]))) + 1,
                        max(0, min(1000, int(1000 * y["y0"]))) + 1,
                        max(0, min(1000, int(1000 * y["x0"]) + int(1000 * y["w"]))) + 1,
                        max(0, min(1000, int(1000 * y["y0"]) + int(1000 * y["h"]))) + 1,
                    )
                    for y in x["ocr_data"]
                ],
            )
            for x in img_data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vqa_sample = self.samples[idx]
        return vqa_sample_2_tensor(
            vqa_sample,
            self.decoder_tokenizer,
            self.max_length,
            os.path.join(self.base_path, self.image_folder, vqa_sample.image_id),
            self.image_processor,
            self.ocr_text_tokenizer,
            self.hash_embed_n_tok,
            self.hash_embed_n_hash,
            self.mode,
        )


if __name__ == "__main__":
    ds = InfoVqaDataset(path="../../data/Info-VQA")
    sample = ds[100]
    print(sample)

    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=CustomDataCollator())
    for batch in dl:
        print(batch)
