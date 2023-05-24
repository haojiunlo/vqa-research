import re
from typing import Any, Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.file_utils import ModelOutput

from src.utils.custom_modeling_gpt_neo import (
    GPTNeoForCausalLM as CustomGPTNeoForCausalLM,
)

# TODO: OCR encoder, custom VisionEncoderDecoderModel that has vit, ocr encoder and bart decoder, \
# custom processor to process ocr output text, input text and image


class VQAModel(torch.nn.Module):
    def __init__(
        self,
        pretrained_img_enc: str,
        pretrained_dec: str,
        pretrained_ocr_enc: str,
        dec_tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.img_encoder = AutoModel.from_pretrained(pretrained_img_enc)
        self.ocr_encoder = AutoModel.from_pretrained(pretrained_ocr_enc)
        # TODO: intergrate with ocr_embedding
        if "gpt-neo" in pretrained_dec.lower():
            self.decoder = CustomGPTNeoForCausalLM.from_pretrained(pretrained_dec)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(pretrained_dec)
        self.decoder_tokenizer = dec_tokenizer
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))

        self.enc_to_dec_proj = torch.nn.Sequential(
            torch.nn.Linear(
                self.ocr_encoder.config.hidden_size
                + self.img_encoder.config.hidden_size,
                self.decoder.config.hidden_size,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.decoder.config.hidden_size,
                self.decoder.config.hidden_size,
            ),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        ocr_text_tensors: torch.Tensor,
        ocr_text_attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        decoder_labels: torch.Tensor = None,
    ):
        img_encoder_outputs = self.img_encoder(image_tensors)
        img_encoder_hidden_states = img_encoder_outputs[0]
        ocr_encoder_outputs = self.ocr_encoder(
            ocr_text_tensors, bbox=bbox, attention_mask=ocr_text_attention_mask
        )
        ocr_encoder_hidden_states = ocr_encoder_outputs[0]

        encoder_hidden_states = torch.concat(
            [img_encoder_hidden_states, ocr_encoder_hidden_states], axis=-1
        )
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=decoder_labels,
        )
        return decoder_outputs

    def inference(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        ocr_text_tensors: torch.Tensor,
        ocr_text_attention_mask: torch.Tensor,
        bbox_tensor: torch.Tensor,
        decoder_tokenizer: PreTrainedTokenizer,
    ):
        img_encoder_hidden_states = self.img_encoder(image_tensors)[0]
        ocr_encoder_hidden_states = self.ocr_encoder(
            ocr_text_tensors, bbox=bbox_tensor, attention_mask=ocr_text_attention_mask
        )[0]

        encoder_hidden_states = torch.concat(
            [img_encoder_hidden_states, ocr_encoder_hidden_states], axis=-1
        )

        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        encoder_outputs = ModelOutput(
            last_hidden_state=encoder_hidden_states, attentions=None
        )

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.unsqueeze(0)
            )
        if len(decoder_input_ids.size()) == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.generate(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            max_length=64,
            early_stopping=True,
            use_cache=True,
            num_beams=1,
            return_dict_in_generate=True,
            output_attentions=False,
        )

        output = {"predictions": list()}
        for seq in decoder_tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(decoder_tokenizer.pad_token, "")
            output["predictions"].append(seq)

        return output


if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer

    DEFAULT_PRETRAINED_BART = "sshleifer/tiny-mbart"
    DEFAULT_PRETRAINED_VIT = "google/vit-base-patch16-224"
    DEFAULT_PRETRAINED_LAYOUTLM = "microsoft/layoutlm-base-uncased"

    # example image
    sample_image_path = "src/models/samples/iphone-14.jpeg"
    im = Image.open(sample_image_path)
    im = im.convert("RGB")

    # init model
    model = VQAModel(
        pretrained_img_enc=DEFAULT_PRETRAINED_VIT,
        pretrained_dec=DEFAULT_PRETRAINED_BART,
        pretrained_ocr_enc=DEFAULT_PRETRAINED_LAYOUTLM,
    )

    seq_length = model.img_encoder.embeddings.patch_embeddings.num_patches + 1

    # init processor and tokenizer
    image_processor = AutoImageProcessor.from_pretrained(DEFAULT_PRETRAINED_VIT)
    decoder_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_BART)
    orc_text_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_LAYOUTLM)

    # prepare inputs
    # decoder input
    question = "What is the product title?"

    # ocr encoder input
    ocr_text = ["Hello", "world"]
    normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
    # Note that one first needs to normalize the bounding boxes to be on a 0-1000 scale. To normalize, you can use the following function:
    # def normalize_bbox(bbox, width, height):
    #     return [
    #         int(1000 * (bbox[0] / width)),
    #         int(1000 * (bbox[1] / height)),
    #         int(1000 * (bbox[2] / width)),
    #         int(1000 * (bbox[3] / height)),
    #     ]

    token_boxes = []
    for word, box in zip(ocr_text, normalized_word_boxes):
        word_tokens = orc_text_tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))

    # add bounding boxes of cls + sep tokens
    token_boxes = (
        [[0, 0, 0, 0]]
        + token_boxes
        + [[1000, 1000, 1000, 1000]] * (seq_length - 1 - len(token_boxes))
    )  # pad to seq_length
    bbox = torch.tensor([token_boxes])

    decoder_input_ids = decoder_tokenizer(
        question, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    ocr_text_input = orc_text_tokenizer(
        " ".join(ocr_text),
        return_tensors="pt",
        max_length=seq_length,
        padding="max_length",
    )

    ocr_text_input_ids = ocr_text_input["input_ids"]
    ocr_text_attention_mask = ocr_text_input["attention_mask"]

    pixel_values = image_processor(im, return_tensors="pt").pixel_values

    # run forward
    output = model(
        pixel_values,
        decoder_input_ids,
        ocr_text_input_ids,
        ocr_text_attention_mask,
        bbox,
    )
    print(output)

    # run inference
    output = model.inference(
        pixel_values,
        decoder_input_ids,
        ocr_text_input_ids,
        ocr_text_attention_mask,
        bbox,
        decoder_tokenizer,
    )
    print(output)
