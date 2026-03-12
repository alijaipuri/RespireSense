from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape

def build_lstm(input_dim):

    model = Sequential()

    model.add(Reshape((input_dim,1),input_shape=(input_dim,)))
    model.add(LSTM(32))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(4,activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
