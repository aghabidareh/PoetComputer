from pathlib import Path

class Config:
    DATA_PATH = Path('data/a.txt')
    PROCESSED_DATA = Path("processed_poems.txt")
    MODEL_NAME = "bolbolzaban/gpt2-persian"
    OUTPUT_DIR = Path("model")

    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 3e-4
    SAVE_STEPS = 1000
    LOGGING_STEPS = 100

    DEFAULT_LINES = 4
    MAX_LENGTH = 400
    TEMPERATURE = 0.7
    TOP_K = 50
    TOP_P = 0.95

config = Config()