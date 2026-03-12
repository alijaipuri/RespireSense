from tensorflow.keras.models import load_model
import numpy as np
from config import *

model = load_model(MODEL_PATH)

def predict_condition(resp,spo2,ph):

    X = np.array([[resp,spo2,ph]])

    pred = model.predict(X).argmax()

    labels = ["Normal","Mild","Severe","Chronic"]

    return labels[pred]
