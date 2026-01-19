import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("house_price_model.pkl")

app = FastAPI(title='House Price Prediction API')


class HouseFeatures(BaseModel):
    avg_area_income: float
    avg_area_house_age: float
    avg_area_number_of_rooms: float
    avg_area_number_of_bedrooms: float
    area_population: float


@app.get("/")
def home():
    return {"message": "House Price Prediction is running"}


@app.post("/predict")
def predict(features: HouseFeatures):
    data = [[
        features.avg_area_income,
        features.avg_area_house_age,
        features.avg_area_number_of_rooms,
        features.avg_area_number_of_bedrooms,
        features.area_population
    ]]
    prediction = model.predict(data)[0]
    return {"predicted_price:", round(prediction, 2)}
