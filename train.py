import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from configs import config
from utils import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

def train():
    tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(config.MODEL_NAME)

    dataset = load_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        overwrite_output_dir=False,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_steps=config.SAVE_STEPS,
        save_total_limit=2,
        logging_steps=config.LOGGING_STEPS,
        learning_rate=config.LR,
        fp16=torch.cuda.is_available(),
        resume_from_checkpoint=True if any((config.OUTPUT_DIR / "checkpoint").glob("*")) else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(config.OUTPUT_DIR)

if __name__ == "__main__":
    train()