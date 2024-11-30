import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

app = FastAPI(title="Ad click prediction")

class AdDetails(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: int
    Area_Income: float
    Daily_Internet_Usage: float
    Ad_topic_line: str
    City: str
    Male: int
    Country: str

with open('./models/final.pkl', 'rb') as f:
    pipe = pickle.load(f)

@app.post("/predict")
def predict(Ad: AdDetails):
    
    data_point = np.array([
        Ad.Daily_Time_Spent_on_Site,
        Ad.Age,
        Ad.Area_Income,
        Ad.Daily_Internet_Usage,
        Ad.Ad_topic_line,
        Ad.City,
        Ad.Male,
        Ad.Country
    ], dtype=object).reshape(1, -1)

    pred = pipe.predict(data_point)
    pred = pred[0]

    return {
        "prediction": int(pred)
    }
