# FastAPI ML Pipeline

API REST construite avec FastAPI qui implemente un pipeline complet de Data Science en 5 phases : nettoyage des donnees, analyse exploratoire, analyse multivariee, modeles ML de base et ML avance (tuning et explicabilite).

## Prerequis

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets)

## Installation

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Lancer le serveur

```bash
uv run uvicorn src.app.api:app --reload
```

L'API est accessible sur `http://127.0.0.1:8000`. La documentation interactive Swagger est disponible sur `http://127.0.0.1:8000/docs`.

## Tests

```bash
# Lancer tous les tests
PYTHONPATH=. uv run pytest

# Lancer les tests d'une phase specifique (1 a 5)
PYTHONPATH=. uv run pytest tests/test_phase1.py

# Lancer le test de bout en bout
PYTHONPATH=. uv run pytest tests/test_end_to_end.py
```

## Architecture

```
src/app/
├── api.py              # Point d'entree FastAPI, montage des routeurs
├── streamlit_app.py    # Interface graphique Streamlit
├── core/
│   ├── dataset.py      # Stockage en memoire (DATASETS, CLEANERS, MODELS)
│   └── schemas.py      # Modeles Pydantic (requetes/reponses)
├── routers/            # Endpoints API (1 routeur par phase)
│   ├── clean.py        # Phase 1 : Nettoyage
│   ├── eda.py          # Phase 2 : Analyse exploratoire
│   ├── mv.py           # Phase 3 : Analyse multivariee
│   ├── ml.py           # Phase 4 : ML de base
│   └── ml2.py          # Phase 5 : ML avance
└── services/           # Logique metier
    ├── cleaning_service.py
    ├── eda_service.py
    ├── mv_service.py
    ├── ml_service.py
    └── ml_advanced_service.py
```

Le projet suit une architecture **3 couches** qui sépare les responsabilités :

```
Requete HTTP → routers/ → services/ → core/
```

- **`routers/`** : Point d'entrée HTTP. Reçoit les requêtes, valide les paramètres via Pydantic, délègue au service correspondant et renvoie la réponse JSON. Aucune logique metier ici.
- **`services/`** : Logique métier. Contient tout le travail réel : nettoyage pandas, entrainement scikit-learn, calcul de metriques, generation de graphiques Plotly. Ne connait ni FastAPI ni HTTP.
- **`core/`** : Données partagées entre toutes les couches. `schemas.py` definit les modeles Pydantic (`APIResponse`, `Meta`, `Result`) et `dataset.py` gere le stockage en memoire (dictionnaires `DATASETS`, `CLEANERS`, `MODELS`).

Cette séparation permet de tester les services sans lancer le serveur, de remplacer le framework web sans toucher à la logique ML, et de localiser rapidement les modifications (bug HTTP → `routers/`, bug ML → `services/`).

Toutes les données sont stockées en mémoire (dictionnaires Python). Chaque ressource (dataset, cleaner, modèle) est identifiée par un UUID. Toutes les réponses suivent un format uniforme `APIResponse` avec les champs : `meta`, `result`, `report`, `artifacts`.

## Les 5 phases

### Phase 1 - Nettoyage (`/clean`)

Génération de jeux de données avec valeurs manquantes, doublons et outliers. Ajustement d'un pipeline de preprocessing (imputation, encodage, clipping) puis transformation.

```bash
# Generer un dataset
curl -X POST http://127.0.0.1:8000/dataset/generate \
  -H "Content-Type: application/json" \
  -d '{"phase": "clean", "seed": 1, "n": 200}'

# Rapport qualite
curl http://127.0.0.1:8000/clean/report/{dataset_id}

# Ajuster le cleaner
curl -X POST http://127.0.0.1:8000/clean/fit \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "params": {}}'

# Transformer
curl -X POST http://127.0.0.1:8000/clean/transform \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "cleaner_id": "{cleaner_id}"}'
```

### Phase 2 - Analyse exploratoire (`/eda`)

Statistiques descriptives, agrégations groupées et visualisations interactives (histogrammes, boxplots, bar charts) retournées en JSON Plotly.

```bash
# Statistiques descriptives
curl -X POST http://127.0.0.1:8000/eda/summary \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}"}'

# Graphiques
curl -X POST http://127.0.0.1:8000/eda/plots \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}"}'
```

### Phase 3 - Analyse multivariee (`/mv`)

Réduction de dimension par ACP, clustering KMeans avec score silhouette, et matrice de corrélation.

```bash
# ACP
curl -X POST http://127.0.0.1:8000/mv/pca/fit_transform \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "n_components": 2}'

# KMeans
curl -X POST http://127.0.0.1:8000/mv/cluster/kmeans \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "n_clusters": 3}'
```

### Phase 4 - ML de base (`/ml`)

Entraînement de modèles (Regression Logistique, Random Forest) avec métriques (Accuracy, Precision, Recall, F1, AUC) et prédictions.

```bash
# Entrainer un modele
curl -X POST http://127.0.0.1:8000/ml/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "target_col": "target", "model_type": "logreg"}'

# Predire
curl -X POST http://127.0.0.1:8000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "model_id": "{model_id}"}'
```

### Phase 5 - ML avance (`/ml2`)

Optimisation d'hyperparametres avec GridSearchCV, importance des features (globale et par permutation), et explicabilite au niveau instance.

```bash
# Tuning
curl -X POST http://127.0.0.1:8000/ml2/tune \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "{dataset_id}", "target_col": "target", "model_type": "rf", "param_grid": {"n_estimators": [5, 10, 50]}}'

# Importance des features
curl http://127.0.0.1:8000/ml2/feature-importance/{model_id}

# Importance par permutation
curl -X POST http://127.0.0.1:8000/ml2/permutation-importance \
  -H "Content-Type: application/json" \
  -d '{"model_id": "{model_id}", "dataset_id": "{dataset_id}", "target_col": "target"}'

# Explication d'une instance
curl -X POST http://127.0.0.1:8000/ml2/explain-instance \
  -H "Content-Type: application/json" \
  -d '{"model_id": "{model_id}", "instance": {"f0": 0.5, "f1": 0.3}}'
```

## Interface Streamlit

Une interface graphique Streamlit permet de parcourir les 5 phases du pipeline sans utiliser curl.

```bash
# Terminal 1 : lancer l'API
uv run uvicorn src.app.api:app --reload

# Terminal 2 : lancer Streamlit
uv run streamlit run src.app.streamlit_app.py
```

L'interface est accessible sur `http://localhost:8501`. La sidebar permet de naviguer entre les phases et de configurer les parametres de generation (seed, nombre d'observations).

## Docker

Docker permet de partager le projet avec un environnement identique, sans installer Python ni les dépendances.

### Prerequis

- [Docker](https://docs.docker.com/get-docker/)

### Build et lancement

```bash
# Construire l'image
docker build -t fastapi-ml-app .

# Lancer le conteneur
docker run -p 8000:8000 fastapi-ml-app
```

L'API est alors accessible sur `http://localhost:8000/docs`.

### Lancer les tests dans le conteneur

```bash
docker run fastapi-ml-app uv run pytest tests/
```

### Partager l'image

```bash
# Exporter l'image dans un fichier .tar
docker save fastapi-ml-app -o fastapi-ml-app.tar

# Le destinataire importe et lance l'image
docker load -i fastapi-ml-app.tar
docker run -p 8000:8000 fastapi-ml-app
```

## Notebook de demonstration

Le fichier `workflow_demo.ipynb` permet de parcourir les 5 phases et de visualiser les resultats (graphiques Plotly) directement dans votre IDE. Lancer le serveur avant d'executer le notebook.

## Technologies

| Librairie | Utilisation |
|-----------|-------------|
| FastAPI | Framework web REST |
| uvicorn | Serveur ASGI |
| pandas | Manipulation de donnees |
| scikit-learn | Modeles ML, preprocessing, metriques |
| plotly | Visualisations interactives (JSON) |
| matplotlib / seaborn | Visualisations statistiques |
| pytest / httpx | Tests |
