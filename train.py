import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.preprocess import load_and_preprocess_data

def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model():
    X, y, tokenizer, max_sequence_len, total_words = load_and_preprocess_data('data/poems.txt')

    model = create_model(total_words, max_sequence_len)

    checkpoint_path = "models/model_checkpoint.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    history = model.fit(X, y, epochs=100, verbose=1, callbacks=[checkpoint, early_stopping])

    model.save('models/final_poetry_model.h5')

if __name__ == "__main__":
    train_model()
