import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional
from configs.poetry_config import PoetryConfig
from .trainer import CustomTrainer
from data.dataset import PersianPoetryDataset


class PoetryGenerator:
    def __init__(self, config: PoetryConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer, self.model = self._initialize_model()

    def _initialize_model(self):
        tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
        model = GPT2LMHeadModel.from_pretrained(self.config.model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        return tokenizer, model

    def train(self, resume: bool = False):
        train_data, val_data = self._load_data()
        trainer = CustomTrainer(self.model, self.tokenizer, self.config)
        trainer.run_training(train_data, val_data, resume)

    def _load_data(self):
        pass

    def generate_poem(self, prompt: str, num_lines: int = 4) -> str:
        pass