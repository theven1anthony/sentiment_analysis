"""
Interface Streamlit pour l'analyse de sentiment Air Paradis.
Permet de tester l'API de prédiction et de donner du feedback.
"""

import os
import streamlit as st
import requests
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Air Paradis - Analyse de Sentiment",
    page_icon="✈️",
    layout="wide"
)

# Configuration de l'API
# Par défaut : serveur de production Azure
# Variable d'environnement API_URL pour override (Docker ou développement local)
default_api_url = os.getenv("API_URL", "https://sentiment-api-at2025.azurewebsites.net")
API_URL = st.sidebar.text_input(
    "URL de l'API",
    value=default_api_url,
    help="URL de base de l'API FastAPI (production Azure par défaut)"
)

# Titre principal
st.title("✈️ Air Paradis - Analyse de Sentiment")
st.markdown("Interface de test pour l'analyse de sentiment des tweets")

# État de l'API
st.sidebar.header("État de l'API")
try:
    response = requests.get(f"{API_URL}/health", timeout=2)
    if response.status_code == 200:
        health_data = response.json()
        st.sidebar.success("✓ API opérationnelle")

        if health_data.get("model_loaded"):
            st.sidebar.info(f"📊 Modèle: {health_data.get('model_type', 'unknown')}")

            # Récupérer les informations du modèle
            try:
                model_info_response = requests.get(f"{API_URL}/model/info", timeout=2)
                if model_info_response.status_code == 200:
                    model_info = model_info_response.json()
                    st.sidebar.metric("F1-Score", f"{model_info.get('f1_score', 0):.4f}")
                    st.sidebar.metric("Accuracy", f"{model_info.get('accuracy', 0):.4f}")
                    st.sidebar.caption(f"Technique: {model_info.get('technique', 'N/A')}")
            except Exception:
                pass
        else:
            st.sidebar.warning("⚠️ Modèle non chargé")
    else:
        st.sidebar.error("✗ API inaccessible")
except Exception as e:
    st.sidebar.error("✗ API inaccessible")
    st.sidebar.caption(f"Erreur: {str(e)}")

# Séparateur
st.markdown("---")

# Section 1: Prédiction
st.header("1️⃣ Analyser un tweet")

col1, col2 = st.columns([2, 1])

with col1:
    # Textarea pour le texte
    text_input = st.text_area(
        "Texte à analyser",
        height=150,
        placeholder="Exemple: I love flying with Air Paradis! Best airline ever!",
        help="Entrez le texte d'un tweet à analyser (en anglais de préférence)"
    )

    # Bouton de prédiction
    predict_button = st.button("🔍 Analyser le sentiment", type="primary", use_container_width=True)

with col2:
    st.info("""
    **Instructions:**

    1. Entrez un texte de tweet
    2. Cliquez sur "Analyser"
    3. Consultez les résultats
    4. Donnez votre feedback si nécessaire
    """)

# Gestion de la prédiction
if predict_button:
    if not text_input or len(text_input.strip()) == 0:
        st.error("⚠️ Veuillez entrer un texte à analyser")
    else:
        try:
            # Appel à l'API de prédiction
            with st.spinner("Analyse en cours..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input},
                    timeout=10
                )

            if response.status_code == 200:
                prediction_data = response.json()

                # Stocker les données de prédiction dans la session
                st.session_state.last_prediction = prediction_data
                st.session_state.last_text = text_input

                # Affichage des résultats
                st.success("✓ Analyse terminée")

                # Metrics en colonnes
                metric_col1, metric_col2, metric_col3 = st.columns(3)

                sentiment = prediction_data.get("sentiment")
                confidence = prediction_data.get("confidence", 0)

                with metric_col1:
                    sentiment_label = "😊 Positif" if sentiment == 1 else "😞 Négatif"
                    sentiment_color = "🟢" if sentiment == 1 else "🔴"
                    st.metric("Sentiment", sentiment_label, delta=sentiment_color)

                with metric_col2:
                    st.metric("Confiance", f"{confidence:.1%}")
                    st.progress(confidence)

                with metric_col3:
                    st.metric("ID Prédiction", prediction_data.get("prediction_id", "N/A"))

                # Détails supplémentaires
                with st.expander("📋 Détails de la prédiction"):
                    st.json(prediction_data)

            else:
                st.error(f"❌ Erreur API: {response.status_code}")
                st.text(response.text)

        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout: L'API met trop de temps à répondre")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Erreur de connexion: Impossible de joindre l'API")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# Séparateur
st.markdown("---")

# Section 2: Feedback
st.header("2️⃣ Donner un feedback")

# Vérifier s'il y a une prédiction en cours
if "last_prediction" in st.session_state and st.session_state.last_prediction:
    prediction_data = st.session_state.last_prediction

    st.info(f"""
    **Prédiction en cours:**
    - Texte: "{st.session_state.last_text[:100]}{'...' if len(st.session_state.last_text) > 100 else ''}"
    - Sentiment prédit: {"Positif" if prediction_data.get("sentiment") == 1 else "Négatif"}
    - Confiance: {prediction_data.get("confidence", 0):.1%}
    """)

    # Formulaire de feedback
    with st.form("feedback_form"):
        st.subheader("La prédiction est-elle correcte ?")

        # Radio button pour le sentiment réel
        actual_sentiment = st.radio(
            "Quel est le sentiment réel de ce texte ?",
            options=[0, 1],
            format_func=lambda x: "😞 Négatif" if x == 0 else "😊 Positif",
            horizontal=True
        )

        # Bouton de soumission
        submit_feedback = st.form_submit_button("📤 Envoyer le feedback", type="primary", use_container_width=True)

        if submit_feedback:
            try:
                # Appel à l'API de feedback
                with st.spinner("Envoi du feedback..."):
                    feedback_response = requests.post(
                        f"{API_URL}/feedback",
                        json={
                            "text": st.session_state.last_text,
                            "predicted_sentiment": prediction_data.get("sentiment"),
                            "actual_sentiment": actual_sentiment,
                            "prediction_id": prediction_data.get("prediction_id"),
                            "timestamp": datetime.now().isoformat()
                        },
                        timeout=10
                    )

                if feedback_response.status_code == 200:
                    feedback_data = feedback_response.json()

                    st.success(f"✓ {feedback_data.get('message', 'Feedback enregistré')}")

                    # Afficher une alerte si déclenchée
                    if feedback_data.get("alert_triggered"):
                        st.error(f"""
                        🚨 **ALERTE DÉCLENCHÉE**

                        {feedback_data.get('misclassified_count', 0)} erreurs détectées en 5 minutes.
                        Le modèle pourrait nécessiter une réévaluation.
                        """)
                    else:
                        misclassified_count = feedback_data.get('misclassified_count', 0)
                        if misclassified_count > 0:
                            st.warning(f"⚠️ {misclassified_count} erreur(s) détectée(s) dans les 5 dernières minutes (seuil: 3)")

                    # Nettoyer la session
                    if "last_prediction" in st.session_state:
                        del st.session_state.last_prediction
                    if "last_text" in st.session_state:
                        del st.session_state.last_text

                    st.rerun()

                else:
                    st.error(f"❌ Erreur API: {feedback_response.status_code}")
                    st.text(feedback_response.text)

            except requests.exceptions.Timeout:
                st.error("⏱️ Timeout: L'API met trop de temps à répondre")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Erreur de connexion: Impossible de joindre l'API")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

else:
    st.info("ℹ️ Aucune prédiction en cours. Analysez d'abord un texte pour pouvoir donner un feedback.")

# Séparateur
st.markdown("---")

# Footer
st.caption("""
**Air Paradis - Analyse de Sentiment** | Projet MLOps OpenClassrooms

Documentation API: [Production Azure](https://sentiment-api-at2025.azurewebsites.net/docs) | [Local](http://localhost:8000/docs)
""")