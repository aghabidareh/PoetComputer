import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import logging

logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "bolbolzaban/gpt2-persian"
CHECKPOINT_DIR = "./checkpoints"

def load_model():
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model.to(device)
    return model, tokenizer

def load_poems(file_path="poems.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        poems = f.read().split("\n")
    return poems

def create_dataset(poems):
    dataset = Dataset.from_dict({"text": poems})
    return dataset

def train_model(dataset, model, tokenizer, resume_from_checkpoint=None):
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        save_strategy="steps",
        report_to="none",
        fp16=True if torch.cuda.is_available() else False
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    logging.info(f"آموزش مدل از {resume_from_checkpoint if resume_from_checkpoint else 'ابتدا'} شروع شد.")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    logging.info("✅ مدل با موفقیت ذخیره شد.")

if __name__ == "__main__":
    model, tokenizer = load_model()
    poems = load_poems()
    dataset = create_dataset(poems)

    last_checkpoint = CHECKPOINT_DIR if os.path.exists(CHECKPOINT_DIR) else None

    train_model(dataset, model, tokenizer, resume_from_checkpoint=last_checkpoint)
