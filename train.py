from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.preprocess import load_and_preprocess_data

def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 1024, input_length=max_sequence_len-1))
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model():
    dataset, tokenizer, max_sequence_len, total_words = load_and_preprocess_data('data/a.txt')

    model = create_model(total_words, max_sequence_len)

    checkpoint_path = "models/model_checkpoint.h5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    history = model.fit(dataset, epochs=50, verbose=1, callbacks=[checkpoint, early_stopping])

    model.save('models/final_poetry_model.h5')
    print("مدل آموزش داده شده و ذخیره شد.")

if __name__ == "__main__":
    train_model()