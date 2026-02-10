"""Interface Streamlit pour l'API FastAPI ML Pipeline."""

import json

import plotly.graph_objects as go
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ML Pipeline", layout="wide")

API_URL = st.sidebar.text_input("URL de l'API", value="http://127.0.0.1:8000")

st.sidebar.markdown("---")
PHASE = st.sidebar.radio(
    "Phase",
    [
        "1 - Nettoyage",
        "2 - EDA",
        "3 - Analyse multivariée",
        "4 - ML de base",
        "5 - ML avancé",
    ],
)

# Mapping phase label -> API phase name
PHASE_API_MAP = {
    "1 - Nettoyage": "clean",
    "2 - EDA": "eda",
    "3 - Analyse multivariée": "mv",
    "4 - ML de base": "ml",
    "5 - ML avancé": "ml2",
}

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset")
SEED = st.sidebar.number_input("Seed", value=42, step=1)
N_OBS = st.sidebar.number_input("Nombre d'observations", value=1000, step=100, min_value=50)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api(method: str, path: str, **kwargs) -> dict | None:
    """Call the API and return parsed JSON, or display an error."""
    url = f"{API_URL}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Impossible de se connecter à l'API ({url}). Vérifiez que le serveur est lancé.")
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        st.error(f"Erreur API {exc.response.status_code} : {detail}")
    except Exception as exc:
        st.error(f"Erreur inattendue : {exc}")
    return None


def show_plotly(fig_json: str | dict):
    """Render a Plotly figure from JSON."""
    if isinstance(fig_json, str):
        fig_json = json.loads(fig_json)
    fig = go.Figure(fig_json)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — Dataset generation (shared across all phases)
# ---------------------------------------------------------------------------

current_phase_api = PHASE_API_MAP[PHASE]

if st.sidebar.button("Générer le dataset"):
    data = api("post", "/dataset/generate", json={"phase": current_phase_api, "seed": SEED, "n": N_OBS})
    if data is not None:
        dataset_id = data["meta"]["dataset_id"]
        st.session_state[f"dataset_id_{current_phase_api}"] = dataset_id
        cols = data["result"]["metrics"]["columns"]
        shape = data["result"]["metrics"]["shape"]
        st.session_state[f"dataset_preview_{current_phase_api}"] = {
            "data": data["result"]["data"],
            "columns": cols,
            "shape": shape,
            "dataset_id": dataset_id,
        }

# Show current dataset info in sidebar
dataset_id = st.session_state.get(f"dataset_id_{current_phase_api}")
if dataset_id:
    preview = st.session_state.get(f"dataset_preview_{current_phase_api}", {})
    shape = preview.get("shape", [])
    shape_str = f"{shape[0]} x {shape[1]}" if shape else ""
    st.sidebar.success(f"Dataset actif\n\n`{dataset_id[:8]}...`\n\n{shape_str}")
else:
    st.sidebar.info("Aucun dataset. Cliquez sur le bouton ci-dessus.")


# ---------------------------------------------------------------------------
# Dataset preview (shown at the top of each phase)
# ---------------------------------------------------------------------------

def show_dataset_preview():
    """Show the dataset preview if available."""
    preview = st.session_state.get(f"dataset_preview_{current_phase_api}")
    if preview:
        with st.expander("Aperçu du dataset (20 premières lignes)"):
            st.dataframe(pd.DataFrame(preview["data"], columns=preview["columns"]))


# ---------------------------------------------------------------------------
# Phase 1 — Nettoyage
# ---------------------------------------------------------------------------


def phase_clean():
    st.header("Phase 1 — Nettoyage des données")

    if not dataset_id:
        st.info("Générez un dataset depuis la sidebar pour commencer.")
        return

    show_dataset_preview()

    # --- Quality report ---
    st.subheader("Rapport qualité")
    if st.button("Afficher le rapport qualité"):
        data = api("get", f"/clean/report/{dataset_id}")
        if data and data.get("report"):
            report = data["report"]
            for section, content in report.items():
                st.markdown(f"**{section}**")
                if isinstance(content, dict):
                    st.json(content)
                else:
                    st.write(content)

    # --- Fit cleaner ---
    st.subheader("Ajuster un cleaner")
    col1, col2, col3 = st.columns(3)
    with col1:
        impute = st.selectbox("Imputation", ["mean", "median", "most_frequent"], index=0)
    with col2:
        outlier = st.selectbox("Outliers", ["clip", "drop", "none"], index=0)
    with col3:
        categorical = st.selectbox("Encodage catégoriel", ["onehot", "label"], index=0)

    if st.button("Fit"):
        params = {
            "impute_strategy": impute,
            "outlier_strategy": outlier,
            "categorical_strategy": categorical,
        }
        data = api("post", "/clean/fit", json={"dataset_id": dataset_id, "params": params})
        if data and data.get("result"):
            cleaner_id = data["result"]["cleaner_id"]
            st.session_state["cleaner_id"] = cleaner_id
            st.success(f"Cleaner ajusté : `{cleaner_id}`")
            if data["result"].get("metrics"):
                st.json(data["result"]["metrics"])

    # --- Transform ---
    st.subheader("Transformer le dataset")
    cleaner_id = st.session_state.get("cleaner_id")
    if not cleaner_id:
        st.info("Ajustez un cleaner d'abord.")
        return

    if st.button("Transform"):
        data = api(
            "post",
            "/clean/transform",
            json={"dataset_id": dataset_id, "cleaner_id": cleaner_id},
        )
        if data:
            new_id = data["meta"]["dataset_id"]
            st.session_state["dataset_id_clean_transformed"] = new_id
            st.success(f"Dataset transformé : `{new_id}`")
            if data.get("result") and data["result"].get("data"):
                st.dataframe(pd.DataFrame(data["result"]["data"]))
            if data.get("report"):
                with st.expander("Rapport après nettoyage"):
                    st.json(data["report"])


# ---------------------------------------------------------------------------
# Phase 2 — EDA
# ---------------------------------------------------------------------------


def phase_eda():
    st.header("Phase 2 — Analyse exploratoire")

    if not dataset_id:
        st.info("Générez un dataset depuis la sidebar pour commencer.")
        return

    show_dataset_preview()

    # --- Summary ---
    st.subheader("Statistiques descriptives")
    if st.button("Calculer les statistiques"):
        data = api("post", "/eda/summary", json={"dataset_id": dataset_id})
        if data and data.get("result") and data["result"].get("data"):
            stats = data["result"]["data"]
            if stats.get("numeric"):
                st.markdown("**Variables numériques**")
                st.dataframe(pd.DataFrame(stats["numeric"]).T)
            if stats.get("categorical"):
                st.markdown("**Variables catégorielles**")
                st.dataframe(pd.DataFrame(stats["categorical"]).T)

    # --- Plots ---
    st.subheader("Visualisations")
    if st.button("Générer les graphiques"):
        data = api("post", "/eda/plots", json={"dataset_id": dataset_id})
        if data and data.get("artifacts"):
            for name, fig_json in data["artifacts"].items():
                st.markdown(f"**{name}**")
                show_plotly(fig_json)

    # --- GroupBy ---
    st.subheader("Agrégation groupée")
    group_col = st.text_input("Colonne de groupement", value="segment")
    agg_input = st.text_input("Agrégations (JSON)", value='{"income": "mean", "age": "mean"}')
    if st.button("GroupBy"):
        try:
            agg = json.loads(agg_input)
        except json.JSONDecodeError:
            st.error("JSON invalide pour les agrégations.")
            return
        data = api(
            "post",
            "/eda/groupby",
            json={"dataset_id": dataset_id, "group_col": group_col, "agg": agg},
        )
        if data and data.get("result") and data["result"].get("data"):
            st.dataframe(pd.DataFrame(data["result"]["data"]))


# ---------------------------------------------------------------------------
# Phase 3 — Analyse multivariée
# ---------------------------------------------------------------------------


def phase_mv():
    st.header("Phase 3 — Analyse multivariée")

    if not dataset_id:
        st.info("Générez un dataset depuis la sidebar pour commencer.")
        return

    show_dataset_preview()

    # --- Correlation matrix ---
    st.subheader("Matrice de corrélation")
    if st.button("Afficher la matrice de corrélation"):
        data = api("get", f"/mv/report/{dataset_id}")
        if data and data.get("result") and data["result"].get("data"):
            corr = data["result"]["data"]["correlation_matrix"]
            df_corr = pd.DataFrame(corr)
            fig = go.Figure(
                data=go.Heatmap(
                    z=df_corr.values,
                    x=df_corr.columns.tolist(),
                    y=df_corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
            )
            fig.update_layout(title="Matrice de corrélation", height=500)
            st.plotly_chart(fig, use_container_width=True)

    # --- PCA ---
    st.subheader("ACP (PCA)")
    n_components = st.slider("Nombre de composantes", 2, 10, 2, key="pca_nc")
    if st.button("Lancer la PCA"):
        data = api(
            "post",
            "/mv/pca/fit_transform",
            json={"dataset_id": dataset_id, "n_components": n_components},
        )
        if data and data.get("result") and data["result"].get("data"):
            pca_data = data["result"]["data"]
            projected = pca_data["projected_data"]
            evr = pca_data["explained_variance_ratio"]

            st.markdown(f"**Variance expliquée** : {[round(v, 4) for v in evr]}")

            if n_components >= 2:
                df_proj = pd.DataFrame(projected, columns=[f"PC{i+1}" for i in range(n_components)])
                fig = go.Figure(
                    data=go.Scatter(x=df_proj["PC1"], y=df_proj["PC2"], mode="markers", marker=dict(size=4))
                )
                fig.update_layout(title="Projection PCA", xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig, use_container_width=True)

    # --- KMeans ---
    st.subheader("KMeans")
    n_clusters = st.slider("Nombre de clusters", 2, 10, 3, key="km_nc")
    if st.button("Lancer KMeans"):
        data = api(
            "post",
            "/mv/cluster/kmeans",
            json={"dataset_id": dataset_id, "n_clusters": n_clusters},
        )
        if data and data.get("result") and data["result"].get("data"):
            km_data = data["result"]["data"]
            labels = km_data["labels"]
            silhouette = km_data["silhouette_score"]
            st.metric("Score silhouette", round(silhouette, 4))

            # Use PCA for visualization
            pca_resp = api(
                "post",
                "/mv/pca/fit_transform",
                json={"dataset_id": dataset_id, "n_components": 2},
            )
            if pca_resp and pca_resp.get("result") and pca_resp["result"].get("data"):
                projected = pca_resp["result"]["data"]["projected_data"]
                df_viz = pd.DataFrame(projected, columns=["PC1", "PC2"])
                df_viz["cluster"] = [str(l) for l in labels]
                fig = go.Figure()
                for c in sorted(df_viz["cluster"].unique()):
                    mask = df_viz["cluster"] == c
                    fig.add_trace(
                        go.Scatter(
                            x=df_viz.loc[mask, "PC1"],
                            y=df_viz.loc[mask, "PC2"],
                            mode="markers",
                            name=f"Cluster {c}",
                            marker=dict(size=4),
                        )
                    )
                fig.update_layout(title="Clusters (projection PCA)", xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Phase 4 — ML de base
# ---------------------------------------------------------------------------


def phase_ml():
    st.header("Phase 4 — ML de base")

    if not dataset_id:
        st.info("Générez un dataset depuis la sidebar pour commencer.")
        return

    show_dataset_preview()

    st.subheader("Entraîner un modèle")
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Type de modèle", ["logreg", "rf"])
    with col2:
        target_col = st.text_input("Colonne cible", value="target", key="ml_target")

    if st.button("Entraîner"):
        data = api(
            "post",
            "/ml/train",
            json={"dataset_id": dataset_id, "target_col": target_col, "model_type": model_type},
        )
        if data and data.get("result"):
            model_id = data["result"]["model_id"]
            st.session_state[f"model_id_{model_type}"] = model_id
            st.success(f"Modèle entraîné : `{model_id}`")
            if data["result"].get("metrics"):
                metrics = data["result"]["metrics"]
                st.dataframe(pd.DataFrame([metrics], index=[model_type]).T.rename(columns={0: "Score"}))

    # --- Compare models ---
    st.subheader("Comparaison des modèles")
    if st.button("Comparer LogReg vs RF"):
        results = {}
        for mt in ["logreg", "rf"]:
            resp = api(
                "post",
                "/ml/train",
                json={"dataset_id": dataset_id, "target_col": target_col, "model_type": mt},
            )
            if resp and resp.get("result") and resp["result"].get("metrics"):
                results[mt] = resp["result"]["metrics"]
                st.session_state[f"model_id_{mt}"] = resp["result"]["model_id"]

        if len(results) == 2:
            df_cmp = pd.DataFrame(results)
            st.dataframe(df_cmp)

            metric_names = list(results["logreg"].keys())
            fig = go.Figure()
            fig.add_trace(
                go.Bar(name="LogReg", x=metric_names, y=[results["logreg"][m] for m in metric_names])
            )
            fig.add_trace(
                go.Bar(name="RF", x=metric_names, y=[results["rf"][m] for m in metric_names])
            )
            fig.update_layout(barmode="group", title="Comparaison des métriques")
            st.plotly_chart(fig, use_container_width=True)

    # --- Predictions ---
    st.subheader("Prédictions")
    pred_model_id = st.text_input(
        "Model ID",
        value=st.session_state.get("model_id_logreg", st.session_state.get("model_id_rf", "")),
        key="ml_pred_mid",
    )
    if pred_model_id and st.button("Prédire"):
        data = api(
            "post",
            "/ml/predict",
            json={"dataset_id": dataset_id, "model_id": pred_model_id},
        )
        if data and data.get("result") and data["result"].get("data"):
            preds = data["result"]["data"]
            st.write(f"**{len(preds['predictions'])} prédictions**")
            df_pred = pd.DataFrame({"prediction": preds["predictions"]})
            if preds.get("probabilities"):
                probs = preds["probabilities"]
                for i in range(len(probs[0])):
                    df_pred[f"proba_class_{i}"] = [p[i] for p in probs]
            st.dataframe(df_pred.head(20))


# ---------------------------------------------------------------------------
# Phase 5 — ML avancé
# ---------------------------------------------------------------------------


def phase_ml2():
    st.header("Phase 5 — ML avancé")

    if not dataset_id:
        st.info("Générez un dataset depuis la sidebar pour commencer.")
        return

    show_dataset_preview()

    # --- Tuning ---
    st.subheader("Tuning d'hyperparamètres")
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Type de modèle", ["logreg", "rf"], key="ml2_mt")
    with col2:
        target_col = st.text_input("Colonne cible", value="target", key="ml2_target")

    if model_type == "logreg":
        default_grid = '{"C": [0.01, 0.1, 1, 10]}'
    else:
        default_grid = '{"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]}'

    param_grid_str = st.text_area("Grille d'hyperparamètres (JSON)", value=default_grid)

    if st.button("Lancer le tuning"):
        try:
            param_grid = json.loads(param_grid_str)
        except json.JSONDecodeError:
            st.error("JSON invalide pour la grille.")
            return

        with st.spinner("Tuning en cours..."):
            data = api(
                "post",
                "/ml2/tune",
                json={
                    "dataset_id": dataset_id,
                    "target_col": target_col,
                    "model_type": model_type,
                    "param_grid": param_grid,
                },
            )
        if data and data.get("result"):
            tuned_id = data["result"]["model_id"]
            st.session_state["tuned_model_id"] = tuned_id
            st.success(f"Meilleur modèle : `{tuned_id}`")
            if data["result"].get("metrics"):
                st.json(data["result"]["metrics"])
            if data["result"].get("data") and data["result"]["data"].get("best_params"):
                st.markdown("**Meilleurs paramètres :**")
                st.json(data["result"]["data"]["best_params"])

    # --- Feature importance ---
    st.subheader("Importance des features")
    fi_model_id = st.text_input("Model ID", value=st.session_state.get("tuned_model_id", ""), key="ml2_fi_mid")

    col_a, col_b = st.columns(2)
    with col_a:
        if fi_model_id and st.button("Feature importance (globale)"):
            data = api("get", f"/ml2/feature-importance/{fi_model_id}")
            if data and data.get("result") and data["result"].get("data"):
                imp = data["result"]["data"]
                df_imp = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).sort_values(
                    "importance", ascending=True
                )
                fig = go.Figure(
                    go.Bar(x=df_imp["importance"], y=df_imp["feature"], orientation="h")
                )
                fig.update_layout(title="Feature Importance (globale)", height=400)
                st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if fi_model_id and dataset_id and st.button("Permutation importance"):
            with st.spinner("Calcul en cours..."):
                data = api(
                    "post",
                    "/ml2/permutation-importance",
                    json={"model_id": fi_model_id, "dataset_id": dataset_id, "target_col": target_col},
                )
            if data and data.get("result") and data["result"].get("data"):
                imp = data["result"]["data"]
                df_imp = pd.DataFrame(list(imp.items()), columns=["feature", "importance"]).sort_values(
                    "importance", ascending=True
                )
                fig = go.Figure(
                    go.Bar(x=df_imp["importance"], y=df_imp["feature"], orientation="h")
                )
                fig.update_layout(title="Permutation Importance", height=400)
                st.plotly_chart(fig, use_container_width=True)

    # --- Explain instance ---
    st.subheader("Explication d'une instance")
    instance_str = st.text_area(
        "Instance (JSON)",
        value='{"f0": 0.5, "f1": -0.3, "f2": 1.0, "f3": 0.0, "f4": 0.8, "f5": -0.5, "f6": 0.2, "f7": 0.1, "f8": -0.7, "f9": 0.4}',
        key="ml2_instance",
    )
    if fi_model_id and st.button("Expliquer"):
        try:
            instance = json.loads(instance_str)
        except json.JSONDecodeError:
            st.error("JSON invalide.")
            return
        data = api(
            "post",
            "/ml2/explain-instance",
            json={"model_id": fi_model_id, "instance": instance},
        )
        if data and data.get("result") and data["result"].get("data"):
            expl = data["result"]["data"]
            st.write(f"**Prédiction** : {expl['prediction']}")
            if expl.get("probabilities"):
                st.write(f"**Probabilités** : {[round(p, 4) for p in expl['probabilities']]}")
            if expl.get("contribution"):
                contrib = expl["contribution"]
                base = contrib.pop("base_value", None)
                if base is not None:
                    st.write(f"**Valeur de base** : {round(base, 4)}")
                df_c = pd.DataFrame(list(contrib.items()), columns=["feature", "contribution"]).sort_values(
                    "contribution"
                )
                colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df_c["contribution"]]
                fig = go.Figure(
                    go.Bar(
                        x=df_c["contribution"],
                        y=df_c["feature"],
                        orientation="h",
                        marker_color=colors,
                    )
                )
                fig.update_layout(title="Contributions par feature", height=400)
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

PHASES = {
    "1 - Nettoyage": phase_clean,
    "2 - EDA": phase_eda,
    "3 - Analyse multivariée": phase_mv,
    "4 - ML de base": phase_ml,
    "5 - ML avancé": phase_ml2,
}

PHASES[PHASE]()
