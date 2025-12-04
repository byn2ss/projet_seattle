import bentoml
import joblib

# Chargement du meilleur modèle entraîné (pipeline sklearn)
# Le fichier best_rf_pipeline.joblib est déjà dans ton dossier Seattle_projet
model = joblib.load("best_rf_pipeline.joblib")

# Sauvegarde du modèle dans le registre BentoML
bento_model = bentoml.sklearn.save_model(
    "energy_rf_pipeline",  # nom logique du modèle (sans :latest)
    model,
)

print(f"✅ Modèle sauvegardé sous le tag : {bento_model.tag}")

