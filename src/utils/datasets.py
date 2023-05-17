import json
import os
from collections import Counter, namedtuple

from torch.utils.data import Dataset

TextVqaSample = namedtuple(
    "TextVqaSample",
    ["question", "answer", "image_id", "img_width", "img_height", "ocr_info"],
)

OcrInfo = namedtuple("OcrInfo", ["word", "w", "h", "x0", "y0", "x1", "y1"])


class TextVqaDataset(Dataset):
    def __init__(self, path: str = "data/TextVQA"):
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
                    OcrInfo(
                        x["word"],
                        int(v["img_width"] * x["w"]),
                        int(v["img_height"] * x["h"]),
                        int(v["img_width"] * x["x0"]),
                        int(v["img_height"] * x["y0"]),
                        int(v["img_width"] * x["x0"]) + int(v["img_width"] * x["w"]),
                        int(v["img_height"] * x["y0"]) + int(v["img_height"] * x["h"]),
                    )
                    for x in imgid2ocr.get(v["image_id"])
                ],
            )
            for v in img_data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> TextVqaSample:
        return self.samples[idx]


if __name__ == "__main__":
    ds = TextVqaDataset(
        path="/Users/charles/Projects/brew/brew-1146-vqa-research/data/TextVQA"
    )
    print(ds[100])
