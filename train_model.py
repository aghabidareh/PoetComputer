import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from data_preparation import prepare_dataset

with open("config/model_config.json", "r") as f:
    model_config = json.load(f)

with open("config/training_config.json", "r") as f:
    training_config = json.load(f)


def train_model():
    dataset = prepare_dataset("poems.txt")
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(model_config["model_name"])

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=model_config["max_length"])

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        save_steps=training_config["save_steps"],
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=training_config["evaluation_steps"],
        logging_dir=training_config["logging_dir"],
        warmup_steps=training_config["warmup_steps"],
        weight_decay=0.01,
        logging_steps=100,
        fp16=training_config["fp16"],
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(100)),
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_config["early_stopping_patience"])]
    )

    trainer.train()
    trainer.save_model(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])


if __name__ == "__main__":
    train_model()
