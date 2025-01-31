from dataclasses import dataclass

@dataclass(frozen=True)
class PoetryConfig:
    model_name: str = 'bolbolzaban/gpt2-persian'
    max_length: int = 512
    temperature: float = 0.85
    top_k: int = 60
    top_p: float = 0.92

    epochs: int = 15
    batch_size: int = 6
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    data_path: str = 'data/a.txt'
    train_ratio: float = 0.92
    checkpoint_dir: str = 'experiments/checkpoints'
    best_model_dir: str = 'experiments/best_model'

    default_lines: int = 4
    max_generation_attempts: int = 3
    seed: int = 42