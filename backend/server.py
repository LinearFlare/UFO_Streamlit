import pickle
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
import uvicorn
import os

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = pickle.load(open(os.path.join(BASE_DIR, "model", "ufo-model.pkl"), "rb"))
countries = ["Australia", "Canada", "Germany", "UK", "US"]


class UFOInput(BaseModel):
    seconds: float
    latitude: float
    longitude: float


@app.get("/")
def index():
    return {"ok": True}


@app.post("/predict/")
async def predict(data: UFOInput):
    features = np.array([[data.seconds, data.latitude, data.longitude]])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    return {
        "prediction": countries[prediction[0]],
        "probabilities": dict(zip(countries, probabilities.tolist())),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
