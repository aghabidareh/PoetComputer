from datasets import load_dataset
from configs import config


def loader(tokenizer):
    dataset = load_dataset("text", data_files={"train": str(config.PROCESSED_DATA)})

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=128)
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset["train"]