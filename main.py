# started 1/4/2026

from fastapi import FastAPI
from pydantic import BaseModel
from mvp_service import MVPService
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()           # create the web app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for local testing)
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
service = MVPService()    # load your model, scalers, and explainer once


class Features(BaseModel):
    temp_max_F: float
    humidity_pct: float
    windspeed_mph: float
    precip_in: float
    ndvi: float
    pop_density: float
    slope: float

@app.post("/predict")
def predict(features: Features):
    return service.predict(features.dict())

@app.post("/explain")
def explain(features: Features):
    return service.explain(features.dict())