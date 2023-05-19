from dataclasses import dataclass

import numpy as np
import torch
import transformers


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
