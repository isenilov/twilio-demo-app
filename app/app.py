from flask import Flask
from flask import request, jsonify
import logging
import torch

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)

app = Flask(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model.to(device)
length = 32


@app.route("/generate/", methods=['GET', 'POST'])
def root():
    prompt_text = str(request.values['Body'])
    print(prompt_text)
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=length,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.2
    )

    generated_sequence = output_sequences[0].tolist()
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    text = text[: None]
    text = "".join(x for x in text if x.isalnum() or x in [' ', ',', '.'])

    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message><Body> {} </Body></Message>
</Response>""".format(text)


if __name__ == "__main__":
    app.run(host="0.0.0.0")