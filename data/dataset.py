import torch
from torch.utils.data import Dataset
import re
from typing import List
from conigs.poetry_config import PoetryConfig

class PersianPoetryDataset(Dataset):
    def __init__(self, tokenizer, poems: List[str], config: PoetryConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = self._process_poems(poems)

    def _process_poems(self, poems: List[str]) -> List[torch.Tensor]:
        return [self._encode_poem(p) for p in poems]

    def _encode_poem(self, poem: str) -> torch.Tensor:
        poem = self._normalize_text(poem)
        return self.tokenizer.encode(
            poem,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors='pt'
        ).squeeze()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return re.sub(r'[ـ\r]', '', text).strip()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {'input_ids': self.examples[idx], 'labels': self.examples[idx]}