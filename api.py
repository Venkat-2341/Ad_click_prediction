import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

app = FastAPI(title="Ad click prediction")

class AdDetails(BaseModel):
    Daily_Time_Spent_on_Site: float
    Age: float
    Area_Income: float
    Daily_Internet_Usage: float
    Ad_topic_line: str
    City: str
    Male: int
    Country: str
    timestamp: str

with open('./models/preprocessors.pkl', 'rb') as f:
    preprocessors = pickle.load(f)
le_country = preprocessors["le_country"]
le_city = preprocessors["le_city"]
le_topic = preprocessors["le_topic"]
scaler = preprocessors["scaler"]


with open('./models/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(Ad: AdDetails):
    
    ad_topic_encoded = le_topic.transform([Ad.Ad_topic_line])[0]
    city_encoded = le_city.transform([Ad.City])[0]
    country_encoded = le_country.transform([Ad.Country])[0]

    timestamp = datetime.strptime(Ad.timestamp, "%Y-%m-%d %H:%M:%S")  
    month = timestamp.month
    day = timestamp.day
    hour = timestamp.hour

    data_point = np.array([[
        Ad.Daily_Time_Spent_on_Site,
        Ad.Age,
        Ad.Area_Income,
        Ad.Daily_Internet_Usage,
        ad_topic_encoded,
        city_encoded,
        Ad.Male,
        country_encoded,
        month,
        day,
        hour
    ]])

    data_point_scaled = scaler.transform(data_point)

    pred = model.predict(data_point_scaled).tolist()
    pred = pred[0]
    print(pred)

    return {
        "prediction": pred
    }
