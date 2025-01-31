from transformers import Trainer, TrainingArguments
from configs.poetry_config import PoetryConfig
from typing import Optional


class CustomTrainer:
    def __init__(self, model, tokenizer, config: PoetryConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def run_training(self, train_data, val_data, resume: bool):
        training_args = self._get_training_args()
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data
        )
        self._handle_resume_training(trainer, resume)

    def _get_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fp16=(self.config.device == 'cuda'),
            logging_dir='logs/',
            save_strategy='epoch',
            evaluation_strategy='epoch',
            load_best_model_at_end=True
        )

    def _handle_resume_training(self, trainer, resume: bool):
        if resume:
            last_checkpoint = self._find_last_checkpoint()
            if last_checkpoint:
                trainer.train(resume_from_checkpoint=last_checkpoint)
                return
        trainer.train()

    def _find_last_checkpoint(self) -> Optional[str]:
        pass