from datasets import Dataset

def load_and_clean_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines

def prepare_dataset(file_path):
    poems = load_and_clean_data(file_path)
    dataset = Dataset.from_dict({"text": poems})
    return dataset

if __name__ == "__main__":
    DATA_FILE = "poems.txt"
    dataset = prepare_dataset(DATA_FILE)
    print(dataset)
