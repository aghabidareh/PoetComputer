import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from configs import config


class PoetryGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(config.OUTPUT_DIR).to(self.device)

    def generate(self, prompt, num_lines=config.DEFAULT_LINES):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=config.MAX_LENGTH,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            top_p=config.TOP_P,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1
        )

        poem = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return self.format_poem(poem, num_lines)

    def format_poem(self, text, num_lines):
        lines = [line.strip() for line in text.split('.') if line.strip()]
        return '\n'.join(lines[:num_lines])


if __name__ == "__main__":

    generator = PoetryGenerator()
    poem = generator.generate("بسی رنج بردم بدین سال سی", 4)
    print("\nشعر تولید شده:\n")
    print(poem)
