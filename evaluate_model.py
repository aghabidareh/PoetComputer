from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = "./models/poetry"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)

    test_texts = [
        "ای دل تو چه کردی که خدا کرد",
        "زندگی زیباست اگر",
        "آسمان پرستاره شب"
    ]

    for text in test_texts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {text}\nGenerated: {generated}\n")


if __name__ == "__main__":
    evaluate_model()
