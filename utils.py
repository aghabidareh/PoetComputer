from transformers import LineByLineTextDataset
from configs import config

def load_dataset(tokenizer):
    return LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(config.PROCESSED_DATA),
        block_size=128
    )