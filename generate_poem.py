import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./models/poetry"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    return tokenizer, model


def generate_poem(start_text, num_lines, tokenizer, model):
    input_ids = tokenizer.encode(start_text, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        input_ids,
        max_length=50 * num_lines,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    poem = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return "\n".join(poem.split("،")[:num_lines])


if __name__ == "__main__":
    tokenizer, model = load_model()
    start_text = input("متن شروع: ")
    num_lines = int(input("تعداد مصراع: "))
    poem = generate_poem(start_text, num_lines, tokenizer, model)
    print("\nGenerated Poem:\n")
    print(poem)
