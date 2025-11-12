# service.py — API BentoML (pipeline sklearn avec prétraitement interne)
import bentoml
import pandas as pd
from typing import Literal, List
from pydantic import BaseModel, Field, validator
from bentoml.io import JSON

# 1) Charger le pipeline (qui inclut OneHotEncoder(handle_unknown="ignore"))
model_ref = bentoml.sklearn.get("energy_rf_pipeline:latest")
model = bentoml.sklearn.load_model(model_ref)

# 2) Schéma d'entrée (validation Pydantic v1)
PrimaryTypes = Literal[
    "Small- and Mid-Sized Office","Other","Warehouse","Large Office","K-12 School",
    "Mixed Use Property","Retail Store","Hotel","Worship Facility","Distribution Center",
    "Supermarket / Grocery Store","Medical Office","Self-Storage Facility","University",
    "Residence Hall","Senior Care Community","Refrigerated Warehouse","Restaurant",
    "Hospital","Laboratory","Office","Low-Rise Multifamily"
]

class EnergyInput(BaseModel):
    PropertyGFATotal: float = Field(..., gt=0, description="Surface totale (ft²)")
    NumberofFloors: int = Field(..., ge=1, le=200)
    YearBuilt: int = Field(..., ge=1800, le=2100)
    PrimaryPropertyType: PrimaryTypes
    HasParking: bool  # true/false côté JSON

    @validator("PropertyGFATotal")
    def check_gfa(cls, v):
        if v > 1e7:
            raise ValueError("Surface trop élevée (max 10M ft²).")
        return v

# 3) Service Bento
svc = bentoml.Service("energy-consumption-predictor")

# 4) Endpoints
@svc.api(input=JSON(pydantic_model=EnergyInput), output=JSON(), name="predict")
def predict(payload: EnergyInput):
    df = pd.DataFrame([payload.dict()])
    # si le pipeline attend 0/1 pour HasParking :
    if isinstance(df.at[0, "HasParking"], (bool,)):
        df["HasParking"] = df["HasParking"].astype(int)
    y = model.predict(df)[0]
    return {"prediction_SiteEnergyUse_kBtu": float(y)}

class EnergyBatch(BaseModel):
    items: List[EnergyInput]

@svc.api(input=JSON(pydantic_model=EnergyBatch), output=JSON(), name="predict_batch")
def predict_batch(payload: EnergyBatch):
    df = pd.DataFrame([p.dict() for p in payload.items])
    if "HasParking" in df.columns:
        df["HasParking"] = df["HasParking"].astype(int)
    y = model.predict(df)
    return {"predictions_SiteEnergyUse_kBtu": [float(v) for v in y]}

@svc.api(input=JSON(), output=JSON(), route="/version", name="version")
def version(_=None):
    return {
        "model_tag": str(model_ref.tag),
        "framework": "sklearn",
        "bentoml_version": bentoml.__version__,
    }
