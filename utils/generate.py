import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_poem(seed_text, num_lines, model, tokenizer, max_sequence_len):
    poem = []
    for _ in range(num_lines):
        line = seed_text
        for _ in range(10):
            token_list = tokenizer.texts_to_sequences([line])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            line += " " + output_word
        poem.append(line.strip())
        seed_text = line
    return "\n".join(poem)