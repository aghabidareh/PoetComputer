import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        poems = file.readlines()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(poems)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for poem in poems:
        token_list = tokenizer.texts_to_sequences([poem])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X = input_sequences[:, :-1]
    y = input_sequences[:, -1]
    y = to_categorical(y, num_classes=total_words)

    return X, y, tokenizer, max_sequence_len, total_words
