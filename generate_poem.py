from tensorflow.keras.models import load_model
from utils.preprocess import load_and_preprocess_data
from utils.generate import generate_poem

def generate_new_poem(seed_text, num_lines=4):
    model = load_model('models/final_poetry_model.h5')

    _, tokenizer, max_sequence_len, _ = load_and_preprocess_data('data/a.txt')

    poem = generate_poem(seed_text, num_lines, model, tokenizer, max_sequence_len)
    print(poem)

if __name__ == "__main__":
    seed_text = "عشق"
    num_lines = 4
    generate_new_poem(seed_text, num_lines)