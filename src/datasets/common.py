from collections import namedtuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from src.models.hash_embedding import HashEmbedding

VqaSample = namedtuple(
    "TextVqaSample",
    ["question", "answer", "image_id", "ocr_info"],
)

OcrInfo = namedtuple("OcrInfo", ["word", "w", "h", "x0", "y0", "x1", "y1"])


def vqa_sample_2_tensor(
    vqa_sample: VqaSample,
    decoder_tokenizer,
    max_len: int,
    img_path: str,
    image_processor,
    hash_embed_n_tok: int,
    hash_embed_n_hash: int,
):
    # Tokenize question and answer using the decoders tokenizer
    question_tensor = decoder_tokenizer(
        vqa_sample.question,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_len,
        padding="max_length",
    ).input_ids.squeeze(0)

    answer_tensor = decoder_tokenizer(
        vqa_sample.answer,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
    ).input_ids.squeeze(0)

    # Preprocess the image
    im = Image.open(img_path)
    im = im.convert("RGB")
    image_tensor = image_processor(im, return_tensors="pt").pixel_values.squeeze(0)
    # Prepare OCR Embedding input
    # Note: not all images have OCR data
    if vqa_sample.ocr_info:
        tok_tensor = HashEmbedding.dataset_prepare_input(
            [x.word for x in vqa_sample.ocr_info],
            n_tok=hash_embed_n_tok,
            n_hash=hash_embed_n_hash,
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
