import numpy as np, pandas as pd, bentoml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

rng = np.random.RandomState(42)
n = 500
primary = ["Small- and Mid-Sized Office","Other","Warehouse","Large Office","K-12 School",
           "Mixed Use Property","Retail Store","Hotel","Worship Facility","Distribution Center",
           "Supermarket / Grocery Store","Medical Office","Self-Storage Facility","University",
           "Residence Hall","Senior Care Community","Refrigerated Warehouse","Restaurant",
           "Hospital","Laboratory","Office","Low-Rise Multifamily"]

df = pd.DataFrame({
    "PropertyGFATotal": rng.randint(2_000, 800_000, size=n),
    "NumberofFloors": rng.randint(1, 30, size=n),
    "YearBuilt": rng.randint(1900, 2016, size=n),
    "PrimaryPropertyType": rng.choice(primary, size=n),
    "HasParking": rng.randint(0, 2, size=n),
})
y = (df["PropertyGFATotal"] * rng.uniform(20, 60, size=n)
     + df["NumberofFloors"]*5_000
     + (2016 - df["YearBuilt"])*3_000
     + df["HasParking"]*50_000
     + rng.normal(0, 1e6, size=n))

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["PrimaryPropertyType"])],
                        remainder="passthrough")

pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
pipe.fit(df, y)

bentoml.sklearn.save_model("energy_rf_pipeline:latest", pipe)
print("✅ OK : modèle sauvegardé avec succès.")
