import re
import torch
import urllib.request

from PIL import Image
from transformers import (DonutProcessor, 
                          VisionEncoderDecoderModel,
                          AutoImageProcessor,
                          AutoTokenizer, 
                          AutoModel,
                          AutoModelForCausalLM)


device = "cuda" if torch.cuda.is_available() else "cpu"


# random mbart model
decoder = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-mbart")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-mbart")


# vit model
vis_encoder=AutoModel.from_pretrained("google/vit-base-patch16-224")
vis_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")


# Vision Encoder Decoder Model
processor = DonutProcessor(
    image_processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"),
    tokenizer=AutoTokenizer.from_pretrained("sshleifer/tiny-mbart")
)

model = VisionEncoderDecoderModel(encoder=AutoModel.from_pretrained("google/vit-base-patch16-224"), decoder=decoder).to(device)


if __name__ == "__main__":
    sample_image_path = "samples/iphone-14.jpeg"
    im = Image.open(sample_image_path)
    im = im.convert("RGB")

    # prepare decoder inputs
    task_prompt = "<s_docvqa><s_question>{}</s_question><s_answer>"
    question = "some question"
    prompt = task_prompt.format(question)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

    pixel_values = processor(im, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
    print(processor.token2json(sequence))
