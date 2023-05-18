import re
from typing import Any, Optional
import torch
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedTokenizer
from transformers.file_utils import ModelOutput

DEFAULT_PRETRAINED_BART = "sshleifer/tiny-mbart"
DEFAULT_PRETRAINED_VIT = "google/vit-base-patch16-224"
DEFAULT_PRETRAINED_LAYOUTLM = "microsoft/layoutlm-base-uncased"

# TODO: OCR encoder, custom VisionEncoderDecoderModel that has vit, ocr encoder and bart decoder, \
# custom processor to process ocr output text, input text and image


class VQAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_encoder = AutoModel.from_pretrained(DEFAULT_PRETRAINED_VIT)
        self.ocr_encoder = AutoModel.from_pretrained(
            DEFAULT_PRETRAINED_LAYOUTLM)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            DEFAULT_PRETRAINED_BART)
        self.enc_to_dec_proj = torch.nn.Sequential(
            torch.nn.Linear(
                self.ocr_encoder.config.hidden_size + self.img_encoder.config.hidden_size,
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
        prompt_tensors: torch.Tensor,
        ocr_text_tensors: torch.Tensor,
        ocr_text_attention_mask: torch.Tensor,
        bbox_tensor: torch.Tensor,
        decoder_tokenizer: PreTrainedTokenizer,
        return_json: bool = True,
        return_attentions: bool = False
    ):
        img_encoder_hidden_states = self.img_encoder(image_tensors)[0]
        ocr_encoder_hidden_states = self.ocr_encoder(
            ocr_text_tensors, bbox=bbox_tensor, attention_mask=ocr_text_attention_mask)[0]

        encoder_hidden_states = torch.concat(
            [img_encoder_hidden_states, ocr_encoder_hidden_states], axis=-1
        )

        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        encoder_outputs = ModelOutput(
            last_hidden_state=encoder_hidden_states, attentions=None)

        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(
                0)
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # get decoder output
        decoder_output = self.decoder.generate(
            input_ids=prompt_tensors,
            encoder_hidden_states=encoder_outputs,
            max_length=self.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[decoder_tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_attentions=False,
        )

        output = {"predictions": list()}
        for seq in decoder_tokenizer.batch_decode(decoder_output.sequences):
            seq = seq.replace(decoder_tokenizer.eos_token, "").replace(
                decoder_tokenizer.pad_token, "")
            # remove first task start token
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()
            if return_json:
                output["predictions"].append(
                    self.token2json(decoder_tokenizer, seq))
            else:
                output["predictions"].append(seq)

        if return_attentions:
            output["attentions"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    def json2token(self, decoder_tokenizer: PreTrainedTokenizer, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.decoder.add_special_tokens(
                            [fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], decoder_tokenizer,
                                          update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(
                    item, decoder_tokenizer, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in decoder_tokenizer.all_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def token2json(self, decoder_tokenizer, tokens, is_inner_value=False):
        """
        Convert a (generated) token seuqnce into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = self.token2json(content, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                leaf in decoder_tokenizer.get_added_vocab()
                                and leaf[0] == "<"
                                and leaf[-2:] == "/>"
                            ):
                                # for categorical special tokens
                                leaf = leaf[1:-2]
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(
                    end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + self.token2json(tokens[6:], is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


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
    image_processor = AutoImageProcessor.from_pretrained(
        DEFAULT_PRETRAINED_VIT)
    decoder_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PRETRAINED_BART)
    orc_text_tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_PRETRAINED_LAYOUTLM)

    # prepare inputs
    # decoder input
    task_prompt = "<s_docvqa><s_question>{}</s_question><s_answer>"
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

    prompt = task_prompt.format(question)
    decoder_input_ids = decoder_tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids

    ocr_text_input = orc_text_tokenizer(" ".join(ocr_text),
                                        return_tensors="pt",
                                        max_length=seq_length,
                                        padding="max_length")

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
    output = model.inference(pixel_values,
                             decoder_input_ids,
                             ocr_text_input_ids,
                             ocr_text_attention_mask,
                             bbox,
                             decoder_tokenizer)
    print(output)
