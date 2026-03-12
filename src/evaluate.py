import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from preprocess import load_data
from config import *

(X_train,X_test,y_train,y_test),_,_ = load_data("../data/respiratory_sample.csv")

model = load_model(MODEL_PATH)

pred = model.predict(X_test).argmax(axis=1)

cm = confusion_matrix(y_test,pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()
