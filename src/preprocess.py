import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):

    df = pd.read_csv(path)

    X = df[["resp_rate","spo2","ph"]]
    y = df["condition"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X,y,test_size=0.2,random_state=42), le, scaler
