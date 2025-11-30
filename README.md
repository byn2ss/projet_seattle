# projet_seattle
## **Contexte du projet**

Je suis **Data Engineer pour la Ville de Seattle**, qui vise la **neutralité carbone d’ici 2050**.
La mairie dispose de relevés de consommation énergétique et d’émissions de CO₂ datant de **2016** pour des **bâtiments non résidentiels** (bureaux, écoles, hôtels, etc.).
L’objectif est de **prédire les consommations totales d’energie et émissions de CO₂** pour les bâtiments **non encore mesurés**, à partir de leurs **caractéristiques structurelles** :
➡️ taille, usage, année de construction, localisation, etc.

- surface totale,
- nombre d’étages,
- année de construction,
- type d’usage,
- localisation, etc.

---

## 🧩 Objectifs du projet

- Réaliser une **analyse exploratoire (EDA)**.
- Nettoyer et préparer les données (gestion des valeurs manquantes, outliers…).
- Faire du **feature engineering** (création de nouvelles variables pertinentes).
- Comparer plusieurs **modèles supervisés** (régression linéaire, Random Forest, SVM…).
- Optimiser le meilleur modèle (GridSearchCV).
- **Exposer le modèle via une API** avec BentoML.

---

## 🗂 Données

- Nombre de bâtiments (après filtrage) : **1 624**
- Nombre de variables : ~**40** (structure, localisation, usage, énergie)

### Principales colonnes utilisées

- Identification / localisation :
  - `OSEBuildingID`, `City`, `Neighborhood`, `Latitude`, `Longitude`, `ZipCode`, `YearBuilt`
- Structure :
  - `PropertyGFATotal`, `NumberofBuildings`, `NumberofFloors`, `PropertyGFAParking`
- Usage :
  - `BuildingType`, `PrimaryPropertyType`, `LargestPropertyUseTypeGFA`
- Cible :
  - `SiteEnergyUse(kBtu)` (et sa version transformée `log_SiteEnergyUse`)

---

## 🧹 Étape 1 – Analyse exploratoire & nettoyage

### Filtrage des bâtiments

- Conservation des **bâtiments non résidentiels** :
  - `NonResidential`, `Nonresidential COS`, `SPS-District K-12`, `Campus`, etc.
- Exclusion des bâtiments à usage **résidentiel** :
  - `Multifamily LR`, `MR`, `HR`, `Residence Hall`, `Senior Care Community`, etc.

> Résultat : **1 624 bâtiments non résidentiels** cohérents avec le périmètre du projet.

### Valeurs manquantes

- Colonnes très manquantes (`YearsENERGYSTARCertified`, uses secondaires…) : conservées dans un premier temps.
- Variables liées directement aux consommations détaillées (`Electricity(kWh)`, etc.) : **supprimées pour éviter le data leakage**.

### Outliers

- Méthode : IQR sur `PropertyGFATotal` et `SiteEnergyUse(kBtu)`.
- Résultat :
  - ~12 % de valeurs extrêmes sur la surface,
  - ~11 % sur la consommation.
- Décision : **conserver les outliers**, car ils représentent de gros bâtiments réalistes (campus, hôpitaux, entrepôts).

---

## 🧠 Étape 2 – Feature engineering

Nouvelles variables créées :

- `BuildingAge` = 2016 - `YearBuilt`  
- `FloorDensity` = `PropertyGFATotal` / `NumberofFloors`
- `HasParking` = 1 si `PropertyGFAParking` > 0, sinon 0
- `GFA_per_building` = `PropertyGFATotal` / `NumberofBuildings`
- `BuildingSizeCategory` = Small / Medium / Large selon `PropertyGFATotal`

Objectif : capturer des informations **structurelles** sans utiliser de données dépendantes des consommations mesurées.

---

## ⚙️ Étape 3 – Préparation des données

- Séparation **train / test** : 80 % / 20 %
- Encodage des variables catégorielles : **One-Hot Encoding**
- Imputation des valeurs manquantes : **médiane** pour les variables numériques
- Mise à l’échelle : `StandardScaler` pour les modèles linéaires et SVM
- Vérification de la qualité des données :
  - absence de `NaN`, `inf`,
  - cohérence des shapes (X_train, X_test).

---

## 🤖 Étape 4 – Modélisation

Modèles testés :

- Régression linéaire
- Random Forest Regressor
- SVM Regressor

Métriques utilisées :

- R²
- MAE
- RMSE

### Meilleur modèle : Random Forest

- **R² (test)** ≈ 0.90–0.96 (selon la version)
- **MAE** faible
- **RMSE** faible

Le modèle explique **plus de 90 % de la variance** de la consommation d’énergie sur le jeu de test.

---

## 🔧 Étape 5 – API avec BentoML

Le meilleur modèle (Random Forest) est sauvegardé avec **BentoML** puis exposé via une API.

### Sauvegarde du modèle

```python
# train_and_save.py
import bentoml
import joblib

model = joblib.load("random_forest_model.pkl")

bento_model = bentoml.sklearn.save_model(
    "energy_rf_pipeline", model
)
