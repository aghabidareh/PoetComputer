import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "./checkpoints"

def load_model():
    model = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(CHECKPOINT_DIR)
    model.to(device)
    return model, tokenizer


def generate_poem(prompt, num_lines=4):
    model, tokenizer = load_model()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    lines = generated_text.split("\n")[:num_lines]
    return "\n".join(lines)

if __name__ == "__main__":
    user_input = input("موضوع یا بیت اولیه: ")
    num_lines = int(input("تعداد مصراع‌ها (پیش‌فرض: 4): ") or 4)
    poem = generate_poem(user_input, num_lines)

    print("\n🎤 شعر تولید شده:\n")
    print(poem)
