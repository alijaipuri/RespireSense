from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def build_ann(input_dim):

    model = Sequential()

    model.add(Dense(32,activation="relu",input_shape=(input_dim,)))
    model.add(Dense(16,activation="relu"))
    model.add(Dense(4,activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
