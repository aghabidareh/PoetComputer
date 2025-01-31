from model.generator import PoetryGenerator
from configs.poetry_config import PoetryConfig
import argparse


def main():
    config = PoetryConfig()
    generator = PoetryGenerator(config)

    poem = generator.generate_poem('بسی رنج بردم بدین سال سی', 4)
    print("\nGenerated Poem:\n")
    print(poem)


if __name__ == "__main__":
    main()