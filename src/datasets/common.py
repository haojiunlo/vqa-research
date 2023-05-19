from collections import namedtuple

import torch
from PIL import Image

from src.models.hash_embedding import HashEmbedding

VqaSample = namedtuple(
    "TextVqaSample",
    ["question", "answer", "image_id", "ocr_info"],
)

OcrInfo = namedtuple("OcrInfo", ["word", "w", "h", "x0", "y0", "x1", "y1"])


def prep_decoder_input_v1(vqa_sample: VqaSample, decoder_tokenizer, max_len: int):
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

    return {"question": question_tensor, "answer": answer_tensor}


def prep_decoder_input_v2(vqa_sample: VqaSample, decoder_tokenizer, max_len: int):
    # v2
    q_ids = decoder_tokenizer(
        vqa_sample.question,
        max_length=max_len,
        truncation=True,
    ).input_ids

    a_ids = decoder_tokenizer(
        vqa_sample.question,
        text_pair=vqa_sample.answer,
        max_length=max_len,
        truncation=True,
    ).input_ids

    input_ids = (
        [decoder_tokenizer.bos_token_id]
        + q_ids
        + [decoder_tokenizer.eos_token_id]
        + a_ids
        + [decoder_tokenizer.eos_token_id]
    )

    # model doesn't need to predict prompt (for VQA)
    labels = [
        id if i > input_ids.index(decoder_tokenizer.eos_token_id) else -100
        for i, id in enumerate(input_ids)
    ]

    return {"input_ids": input_ids, "labels": labels}


def prepare_ocr_input_v1(
    vqa_sample: VqaSample, hash_embed_n_tok: int, hash_embed_n_hash: int
):
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
        "ocr_tok": tok_tensor,
        "ocr_x0": x0_tensor,
        "ocr_y0": y0_tensor,
        "ocr_x1": x1_tensor,
        "ocr_y1": y1_tensor,
        "ocr_w": w_tensor,
        "ocr_h": h_tensor,
    }


def prepare_ocr_input_v2(vqa_sample: VqaSample, ocr_text_tokenizer):
    ocr_text = [x.word for x in vqa_sample.ocr_info]
    normalized_word_boxes = [[x.x0, x.y0, x.x1, x.y1] for x in vqa_sample.ocr_info]
    token_boxes = []
    for word, box in zip(ocr_text, normalized_word_boxes):
        word_tokens = ocr_text_tokenizer.tokenize(word)
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

    ocr_text_input = ocr_text_tokenizer(
        " ".join(ocr_text),
        return_tensors="pt",
        truncation=True,
        max_length=197,  # TODO -- don't hardcode this
        padding="max_length",
    )
    ocr_text_input_ids = ocr_text_input["input_ids"].squeeze(0)
    ocr_text_attention_mask = ocr_text_input["attention_mask"].squeeze(0)

    return {
        "ocr_text_tensor": ocr_text_input_ids,
        "ocr_text_attention_mask": ocr_text_attention_mask,
        "ocr_bbox": bbox,
    }


def prepare_image_input(img_path: str, image_processor):
    im = Image.open(img_path)
    im = im.convert("RGB")
    image_tensor = image_processor(im, return_tensors="pt").pixel_values.squeeze(0)
    return {"image": image_tensor}


def vqa_sample_2_tensor(
    vqa_sample: VqaSample,
    # Decoder
    decoder_tokenizer,
    max_len: int,
    # Img
    img_path: str,
    image_processor,
    # OCR1
    ocr_text_tokenizer,
    # OCR2
    hash_embed_n_tok: int,
    hash_embed_n_hash: int,
):
    # TODO -- redesign the way this works
    # Idea: each model makes their own `prepare_input` function
    # Tokenize question and answer using the decoders tokenizer
    decoder_input = prep_decoder_input_v1(vqa_sample, decoder_tokenizer, max_len)

    # Preprocess the image
    image_input = prepare_image_input(img_path, image_processor)

    # FIXME -- collate functions depend on which ocr you use
    if ocr_text_tokenizer:
        ocr_input = prepare_ocr_input_v2(vqa_sample, decoder_tokenizer)
    else:
        ocr_input = prepare_ocr_input_v1(
            vqa_sample, hash_embed_n_tok, hash_embed_n_hash
        )

    return {**decoder_input, **image_input, **ocr_input}
