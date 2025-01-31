import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def load_and_preprocess_data(file_path, num_words=5000):
    with open(file_path, 'r', encoding='utf-8') as file:
        poems = file.readlines()

    poems = [poem.strip() for poem in poems if poem.strip()]

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(poems)
    total_words = min(len(tokenizer.word_index) + 1, num_words)

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
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(128).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, tokenizer, max_sequence_len, total_words