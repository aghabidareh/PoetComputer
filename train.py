from model.generator import PoetryGenerator
from configs.poetry_config import PoetryConfig
from utils.logger import setup_logger


def main():
    setup_logger()
    config = PoetryConfig()
    generator = PoetryGenerator(config)

    try:
        generator.train(resume=True)
    except KeyboardInterrupt:
        print("\nTraining saved. Safe to exit.")


if __name__ == "__main__":
    main()