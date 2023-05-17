from typing import Optional

import PIL
import torch
from transformers import AutoModel, AutoModelForCausalLM

DEFAULT_PRETRAINED_BART = "sshleifer/tiny-mbart"
DEFAULT_PRETRAINED_VIT = "google/vit-base-patch16-224"
DEFAULT_PRETRAINED_BERT = "distilbert-base-multilingual-cased"


# TODO: OCR encoder, custom VisionEncoderDecoderModel that has vit, ocr encoder and bart decoder, \
# custom processor to process ocr output text, input text and image


class VQAModel(torch.nn.Module):
    r"""
    Donut: an E2E OCR-free Document Understanding Transformer.
    The encoder maps an input document image into a set of embeddings,
    the decoder predicts a desired token sequence, that can be converted to a structured format,
    given a prompt and the encoder output embeddings
    """

    def __init__(self):
        super().__init__()
        self.img_encoder = AutoModel.from_pretrained(DEFAULT_PRETRAINED_VIT)
        self.ocr_encoder = AutoModel.from_pretrained(DEFAULT_PRETRAINED_BERT)
        self.decoder = AutoModelForCausalLM.from_pretrained(DEFAULT_PRETRAINED_BART)
        self.enc_to_dec_proj = torch.nn.Sequential(
            torch.nn.Linear(
                self.ocr_encoder.config.dim + self.img_encoder.config.hidden_size,
                self.decoder.config.hidden_size,
            ),
            torch.nn.ReLU(),
        )

    def forward(
        self,
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        ocr_text_tensors: torch.Tensor,
        decoder_labels: torch.Tensor = None,
    ):
        """
        Calculate a loss given an input image and a desired token sequence,
        the model will be trained in a teacher-forcing manner

        Args:
            image_tensors: (batch_size, num_channels, height, width)
            decoder_input_ids: (batch_size, sequence_length, embedding_dim)
            decode_labels: (batch_size, sequence_length)
        """
        img_encoder_outputs = self.img_encoder(image_tensors)
        img_encoder_hidden_states = img_encoder_outputs[0]
        ocr_encoder_outputs = self.ocr_encoder(ocr_text_tensors)
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


if __name__ == "__main__":
    from PIL import Image
    from transformers import AutoImageProcessor, AutoTokenizer

    # example image
    sample_image_path = "samples/iphone-14.jpeg"
    im = Image.open(sample_image_path)
    im = im.convert("RGB")

    # init model
    model = VQAModel()

    seq_length = model.img_encoder.embeddings.patch_embeddings.num_patches + 1

    # init processor and tokenizer
    image_processor = AutoImageProcessor.from_pretrained(DEFAULT_PRETRAINED_VIT)
    decoder_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_BART)
    orc_text_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_BERT)

    # prepare inputs
    task_prompt = "<s_docvqa><s_question>{}</s_question><s_answer>"
    question = "What is the product title?"
    ocr_text = "ocr text here"

    prompt = task_prompt.format(question)
    decoder_input_ids = decoder_tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=seq_length,  # vit patches num
        padding="max_length",
    ).input_ids

    ocr_text_input_ids = orc_text_tokenizer(
        ocr_text,
        return_tensors="pt",
        max_length=seq_length,  # vit patches num
        padding="max_length",
    ).input_ids

    pixel_values = image_processor(im, return_tensors="pt").pixel_values

    # run model
    output = model(pixel_values, decoder_input_ids, ocr_text_input_ids)
    print(output)
