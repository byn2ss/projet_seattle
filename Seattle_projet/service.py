# service.py — API pour prédire la consommation énergétique (BentoML 1.3.x + Pydantic v1)
import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel, Field, validator
from typing import Literal, List

# Charger le modèle (pipeline sklearn) + ordre des features (si tu l'as sauvegardé dans custom_objects)
model_ref = bentoml.sklearn.get("energy_rf_pipeline:latest")
model = bentoml.sklearn.load_model(model_ref)
feature_order = getattr(model_ref, "custom_objects", {}).get("feature_order", None)

# Liste des types autorisés (à adapter si besoin)
PrimaryTypes = Literal[
    "Small- and Mid-Sized Office", "Other", "Warehouse", "Large Office", "K-12 School",
    "Mixed Use Property", "Retail Store", "Hotel", "Worship Facility", "Distribution Center",
    "Supermarket / Grocery Store", "Medical Office", "Self-Storage Facility", "University",
    "Residence Hall", "Senior Care Community", "Refrigerated Warehouse", "Restaurant",
    "Hospital", "Laboratory", "Office", "Low-Rise Multifamily"
]

class EnergyInput(BaseModel):
    PropertyGFATotal: float = Field(..., gt=0)
    NumberofFloors: int = Field(..., ge=1, le=200)
    YearBuilt: int = Field(..., ge=1800, le=2016)
    PrimaryPropertyType: PrimaryTypes
    HasParking: bool

    @validator("PropertyGFATotal")
    def check_gfa(cls, v):
        if v > 1e7:
            raise ValueError("Surface trop élevée (max 10M ft²).")
        return v

svc = bentoml.Service("energy-consumption-predictor")

def _apply_primary_type_ohe(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """Active la dummy 'PrimaryPropertyType_*' attendue si le modèle a vu ces colonnes en train."""
    if feature_order is None:
        return df  # le pipeline sklearn gère tout (OneHotEncoder dans le modèle)
    raw = str(df.at[0, "PrimaryPropertyType"])
    candidate_dummy = f"PrimaryPropertyType_{raw}"
    other_dummy = "PrimaryPropertyType_Other"
    primary_dummies = [c for c in feature_order if c.startswith("PrimaryPropertyType_")]
    for col in primary_dummies:
        df[col] = 0
    if candidate_dummy in primary_dummies:
        df[candidate_dummy] = 1
    elif other_dummy in primary_dummies:
        df[other_dummy] = 1
    if "PrimaryPropertyType" in df.columns:
        df = df.drop(columns=["PrimaryPropertyType"])
    return df

@svc.api(input=JSON(pydantic_model=EnergyInput), output=JSON(), name="predict")
def predict(payload: EnergyInput):
    base = pd.DataFrame([payload.dict()])
    if isinstance(base.at[0, "HasParking"], (bool,)):
        base["HasParking"] = base["HasParking"].astype(int)
    if feature_order is not None:
        base = _apply_primary_type_ohe(base, feature_order)
        row = base.reindex(columns=feature_order, fill_value=0)
    else:
        row = base  # pipeline gère l'encodage
    y_pred = model.predict(row)[0]
    try:
        y_pred = float(y_pred)
    except Exception:
        pass
    return {"prediction_SiteEnergyUse_kBtu": y_pred}

@svc.api(input=JSON(), output=JSON(), route="/version", name="version")
def version(_=None):
    return {
        "model_tag": str(model_ref.tag),
        "framework": "sklearn",
        "bentoml_version": bentoml.__version__,
    }

class EnergyBatch(BaseModel):
    items: List[EnergyInput]

@svc.api(input=JSON(pydantic_model=EnergyBatch), output=JSON(), name="predict_batch")
def predict_batch(payload: EnergyBatch):
    df = pd.DataFrame([p.dict() for p in payload.items])
    if "HasParking" in df.columns:
        df["HasParking"] = df["HasParking"].astype(int)
    if feature_order is not None:
        # appliquer OHE à chaque ligne si besoin
        out = []
        for i in range(len(df)):
            r = _apply_primary_type_ohe(df.iloc[[i]].copy(), feature_order)
            r = r.reindex(columns=feature_order, fill_value=0)
            out.append(r)
        X = pd.concat(out, ignore_index=True)
    else:
        X = df
    y = model.predict(X)
    return {"predictions_SiteEnergyUse_kBtu": [float(v) for v in y]}
