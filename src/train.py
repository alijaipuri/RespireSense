from preprocess import load_data
from ann_model import build_ann
from lstm_model import build_lstm
from config import *
import tensorflow as tf

(X_train,X_test,y_train,y_test),le,scaler = load_data("../data/respiratory_sample.csv")

print("Training ANN...")
ann = build_ann(X_train.shape[1])
ann.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE)

print("Training LSTM...")
lstm = build_lstm(X_train.shape[1])
lstm.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE)

lstm.save(MODEL_PATH)

print("Model Saved")
