import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import bentoml

# 1️⃣ Charger les données
data = pd.read_csv("2016_Building_Energy_Benchmarking.csv")
# 1) Charger le CSV
data = pd.read_csv("2016_Building_Energy_Benchmarking.csv")

# 2) Créer HasParking si absent (binaire: 1 si parking > 0, sinon 0)
if "HasParking" not in data.columns:
    # PropertyGFAParking existe dans le dataset ; on le convertit en indicateur
    data["HasParking"] = (data["PropertyGFAParking"].fillna(0) > 0).astype(int)

# 3) (optionnel mais recommandé) Nettoyer la cible avant de continuer
data = data[pd.notna(data["SiteEnergyUse(kBtu)"])]

cols_to_keep = [
    "PropertyGFATotal",
    "NumberofFloors",
    "YearBuilt",
    "PrimaryPropertyType",
    "HasParking",
    "SiteEnergyUse(kBtu)"  # <- target
]
data = data[cols_to_keep]

# 2️⃣ Nettoyer les données
data = data[[
    "PropertyGFATotal",
    "NumberofFloors",
    "YearBuilt",
    "PrimaryPropertyType",
    "HasParking",
    "SiteEnergyUse(kBtu)"
]].dropna()

# 3️⃣ Encodage des variables catégorielles
data = pd.get_dummies(data, columns=["PrimaryPropertyType"], drop_first=True)

# 4️⃣ Séparer les variables explicatives et la cible
X = data.drop("SiteEnergyUse(kBtu)", axis=1)
y = data["SiteEnergyUse(kBtu)"]

# 5️⃣ Diviser les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Entraîner le modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7️⃣ Sauvegarder le modèle avec BentoML
bentoml.sklearn.save_model(
    "energy_rf_pipeline",
    model,
    custom_objects={"feature_order": list(X.columns)}
)

print("✅ Modèle sauvegardé avec succès !")

