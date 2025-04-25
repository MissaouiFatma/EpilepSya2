import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go

# ---------- AUTH ----------
import json

USER_DB = "users.json"
PREDICTIONS_LOG = "predictions_log.csv"

if not os.path.exists(USER_DB):
    with open(USER_DB, "w") as f:
        json.dump({}, f)

if not os.path.exists(PREDICTIONS_LOG):
    pd.DataFrame(columns=["Username", "File", "Prediction"]).to_csv(PREDICTIONS_LOG, index=False)

st.set_page_config(layout="wide", page_title="EpilepSya App")

# ---------- STYLE ----------
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
    background-image: linear-gradient(to right, #8360c3, #2ebf91);
    background-size: cover;
}}
.sidebar .sidebar-content {{
    background-color: rgba(0, 0, 0, 0.3);
}}
.step-done {{
    background-color: #28a745;
    color: white;
    padding: 0.5rem;
    border-radius: 10px;
    text-align: center;
    flex: 1;
}}
.step-pending {{
    background-color: #ffc107;
    color: black;
    padding: 0.5rem;
    border-radius: 10px;
    text-align: center;
    flex: 1;
}}
.steps-container {{
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- SESSION ----------
if "username" not in st.session_state:
    st.session_state.username = None


# ---------- AUTH FUNCTIONS ----------
def register_user(username, password):
    with open(USER_DB, "r") as f:
        users = json.load(f)
    if username in users:
        return False
    users[username] = password
    with open(USER_DB, "w") as f:
        json.dump(users, f)
    return True


def login_user(username, password):
    with open(USER_DB, "r") as f:
        users = json.load(f)
    return users.get(username) == password


# ---------- SIDEBAR ----------
st.sidebar.title("🧠 EpilepSya")
selection = st.sidebar.radio("Navigation", ["Accueil", "Enregistrement", "Entraînement", "Prédiction"])

# ---------- ACCUEIL ----------
if selection == "Accueil":
    st.title("Bienvenue sur EpilepSya")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Créer un compte")
        new_user = st.text_input("Nom d'utilisateur")
        new_pass = st.text_input("Mot de passe", type="password")
        if st.button("Créer un compte"):
            if register_user(new_user, new_pass):
                st.success("Compte créé avec succès.")
            else:
                st.error("Nom d'utilisateur déjà existant.")
    with col2:
        st.subheader("Log in")
        user = st.text_input("Nom d'utilisateur (connexion)")
        pwd = st.text_input("Mot de passe (connexion)", type="password")
        if st.button("Connexion"):
            if login_user(user, pwd):
                st.session_state.username = user
                st.success("Connecté avec succès.")
            else:
                st.error("Identifiants incorrects.")

# ---------- ENREGISTREMENT ----------
elif selection == "Enregistrement":
    st.title("📋 Historique des prédictions")
    df_log = pd.read_csv(PREDICTIONS_LOG)
    st.dataframe(df_log)

# ---------- ENTRAINEMENT ----------
elif selection == "Entraînement":
    if not st.session_state.username:
        st.warning("Veuillez vous connecter pour accéder à cette section.")
    else:
        st.title("⚙️ Entraînement et élection du meilleur modèle")
        file = st.file_uploader("Uploader un CSV EEG brut", type=["csv"])

        if file:
            step_state = {k: "done" for k in
                          ["preprocessing", "features", "training", "evaluation", "selection", "save"]}

            st.markdown("""
                <div class='steps-container'>
                    <div class='step-{0}'>1. Prétraitement</div>
                    <div class='step-{1}'>2. Extraction des caractéristiques</div>
                    <div class='step-{2}'>3. Entraînement</div>
                    <div class='step-{3}'>4. Évaluation</div>
                    <div class='step-{4}'>5. Élection</div>
                    <div class='step-{5}'>6. Sauvegarde</div>
                </div>
            """.format(*step_state.values()), unsafe_allow_html=True)

            data = pd.read_csv(file)
            if 'Classe' in data.columns and 'Filename' in data.columns:
                raw_data = data.drop(columns=['Classe', 'Filename'])
                labels = data['Classe']
            else:
                st.error("Le fichier CSV doit contenir les colonnes 'classe' et 'filename'.")
                st.stop()


            def extract_features(signal):
                diff = np.diff(signal)
                return [
                    np.mean(diff),
                    np.std(diff),
                    np.min(diff),
                    np.max(diff),
                    np.median(diff),
                    kurtosis(diff),
                    skew(diff)
                ]


            features = raw_data.apply(extract_features, axis=1, result_type='expand')
            features.columns = ["mean", "std", "min", "max", "median", "kurtosis", "skewness"]

            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "RandomForest": RandomForestClassifier(),
                "SVM": SVC(),
                "DecisionTree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier()
            }

            metrics = {}
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                metrics[name] = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }

            # Sélection du meilleur modèle basé sur la précision
            best_model_name = max(metrics, key=lambda k: metrics[k]['precision'])
            best_model = models[best_model_name]
            best_precision = metrics[best_model_name]['precision']

            # Sauvegarde du meilleur modèle
            joblib.dump((best_model, scaler), "best_model.pkl")

            st.success(f"Modèle élu : {best_model_name} avec une précision de {best_precision:.4f} sauvegardé.")

            # Affichage des performances des modèles
            st.subheader("📊 Comparaison des modèles")
            fig = go.Figure()
            for metric in ["accuracy", "precision", "recall", "f1"]:
                fig.add_trace(go.Bar(
                    x=list(metrics.keys()),
                    y=[metrics[m][metric] for m in metrics],
                    name=metric
                ))
            fig.update_layout(
                barmode='group',
                xaxis_title="Modèle",
                yaxis_title="Score",
                title="Performance des modèles",
                legend_title="Métrique"
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------- PREDICTION ----------
elif selection == "Prédiction":
    if not st.session_state.username:
        st.warning("Veuillez vous connecter pour accéder à cette section.")
    else:
        st.title("🔮 Prédiction à partir d'un fichier CSV")
        pred_file = st.file_uploader("Uploader le fichier CSV pour la prédiction", type=["csv"])
        if pred_file and os.path.exists("best_model.pkl"):
            model, scaler = joblib.load("best_model.pkl")
            pred_data = pd.read_csv(pred_file)
            try:
                raw_pred = pred_data.drop(columns=['Classe', 'Filename'])
            except:
                raw_pred = pred_data


            def extract_features(signal):
                diff = np.diff(signal)
                return [
                    np.mean(diff),
                    np.std(diff),
                    np.min(diff),
                    np.max(diff),
                    np.median(diff),
                    kurtosis(diff),
                    skew(diff)
                ]


            X_pred = raw_pred.apply(extract_features, axis=1, result_type='expand')
            X_pred.columns = ["mean", "std", "min", "max", "median", "kurtosis", "skewness"]
            X_pred_scaled = scaler.transform(X_pred)
            predictions = model.predict(X_pred_scaled)

            st.subheader("🧾 Résultats ligne par ligne")
            interpretation = {
                "Ep": "Épileptique",
                "Hp": "Sain",
                "Sp": "Suspect"
            }
            results = []
            for i, pred in enumerate(predictions):
                label = interpretation.get(pred, pred)
                st.write(f"Ligne {i + 1} : {label}")
                results.append({
                    "Username": st.session_state.username,
                    "File": pred_file.name,
                    "Prediction": label
                })

            pd.DataFrame(results).to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False)

            st.subheader("📄 Données analysées")
            st.dataframe(pred_data.head())
