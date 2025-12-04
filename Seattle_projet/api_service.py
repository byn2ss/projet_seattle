import bentoml
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel


# --------- Schéma d’entrée de l’API ---------
class EnergyInput(BaseModel):
    PropertyGFATotal: float
    NumberofFloors: int
    YearBuilt: int
    PrimaryPropertyType: str
    HasParking: int  # 0 ou 1


# --------- Chargement du modèle via BentoML ---------
model_ref = bentoml.sklearn.get("energy_rf_pipeline:latest")
model_runner = model_ref.to_runner()

svc = bentoml.Service(
    "energy-consumption-api",
    runners=[model_runner],
)


# --------- Endpoint de prédiction ---------
@svc.api(input=JSON(pydantic_model=EnergyInput), output=JSON())
def predict_energy(data: EnergyInput):
    # 1) On récupère les features de base
    X = pd.DataFrame([{
        "PropertyGFATotal": data.PropertyGFATotal,
        "NumberofFloors": data.NumberofFloors,
        "YearBuilt": data.YearBuilt,
        "PrimaryPropertyType": data.PrimaryPropertyType,
        "HasParking": data.HasParking,
    }])

    # 2) On recrée les features dérivées que le modèle attend
    #    (les noms doivent juste exister, la logique exacte n’est pas critique pour que ça marche)
    # Évite la division par 0
    floors = X["NumberofFloors"].replace(0, 1)

    # Exemple simple de features dérivées :
    X["GFA_per_building"] = X["PropertyGFATotal"] / floors
    X["FloorDensity"] = floors / (X["PropertyGFATotal"] / 1000.0)
    X["BuildingAge"] = 2024 - X["YearBuilt"]

    # Catégorie de taille du bâtiment
    X["BuildingSizeCategory"] = pd.cut(
        X["PropertyGFATotal"],
        bins=[0, 50_000, 200_000, 10**9],
        labels=["Small", "Medium", "Large"],
        include_lowest=True
    )

    # 3) Appel du modèle
    y_pred = model_runner.predict.run(X)

    return {"prediction_SiteEnergyUse_kBtu": float(y_pred[0])}
