import os

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


app = Flask(__name__)


TOKENIZER = AutoTokenizer.from_pretrained("hugosousa/temp", token=HF_TOKEN)
MODEL = AutoModelForCausalLM.from_pretrained(
    "hugosousa/temp", device_map="cuda", torch_dtype=torch.bfloat16, token=HF_TOKEN
)


def translate_text(text):
    chat = [
        {
            "role": "system",
            "content": "You are a translator from English to European Portuguese",
        },
        {
            "role": "user",
            "content": f"Translate this text from English to European Portuguese: {text}",
        },
    ]

    input_ids = TOKENIZER.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        max_length=1024,
    ).to(MODEL.device)

    output_ids = MODEL.generate(
        input_ids,
        max_length=1024,
        num_return_sequences=1,
        pad_token_id=TOKENIZER.eos_token_id,
    )

    generated_ids = output_ids[0, input_ids.shape[1] :]
    print(generated_ids.shape)
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    input_text = data["text"]
    # Your translation model logic here
    translated_text = translate_text(input_text)
    return jsonify({"translated_text": translated_text})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
