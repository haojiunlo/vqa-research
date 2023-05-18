import json
import os

from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer

from src.datasets.common import OcrInfo, VqaSample, collate_batch, vqa_sample_2_tensor


class InfoVqaDataset(Dataset):
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

        with open(os.path.join(path, "infographicVQA_train_v1.0.json"), "r") as f:
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
            with open(
                os.path.join(
                    path, "infographicVQA_train_v1.0_ocr_outputs", item["ocr_fn"]
                )
            ) as f:
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
                        int(1000 * y["x0"]) + 1,
                        int(1000 * y["y0"]) + 1,
                        int(1000 * y["x0"]) + int(1000 * y["w"]) + 1,
                        int(1000 * y["y0"]) + int(1000 * y["h"]) + 1,
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
            os.path.join(
                self.base_path, "infographicVQA_train_v1.0_images", vqa_sample.image_id
            ),
            self.image_processor,
            self.hash_embed_n_tok,
            self.hash_embed_n_hash,
        )


if __name__ == "__main__":
    ds = InfoVqaDataset(path="../../data/Info-VQA")
    sample = ds[100]
    print(sample)

    dl = DataLoader(ds, batch_size=5, shuffle=True, collate_fn=collate_batch)
    for batch in dl:
        print(batch)
