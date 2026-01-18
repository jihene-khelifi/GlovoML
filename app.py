import streamlit as st
import pandas as pd
from sklearn import joblib
import time

# --- 1. Configuration de la page ---
st.set_page_config(
    page_title="Glovo Tunisia Predictor",
    page_icon="🛵",
    layout="centered"
)

# --- 2. Style CSS pour le look Glovo ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button {
        background-color: #FFC244;
        color: #000000;
        border-radius: 20px;
        font-weight: bold;
        width: 100%;
    }
    .stHeader { color: #00A082; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. En-tête ---
st.image("https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Glovo_logo.svg/1200px-Glovo_logo.svg.png", width=150)
st.title("🛵 Prédiction de Livraison - Glovo")
st.markdown("### Est-ce que la commande sera en retard ?")
st.write("Cet outil utilise l'intelligence artificielle (Random Forest) pour estimer le statut de livraison.")

# --- 4. Chargement du modèle ---
@st.cache_resource # Pour éviter de recharger le modèle à chaque clic
def load_model():
    return joblib.load('glovo_model.pkl')

try:
    model = load_model()
except:
    st.error("❌ Erreur : Le fichier 'glovo_model.pkl' est introuvable. Assurez-vous de l'avoir généré dans votre Notebook.")
    st.stop()

# --- 5. Interface de saisie ---
st.sidebar.header("📋 Détails de la commande")
distance = st.sidebar.slider("Distance (en km)", 0.1, 20.0, 3.5)
montant = st.sidebar.number_input("Montant Total (DT)", min_value=1.0, max_value=1000.0, value=45.0)

# --- 6. Logique de Prédiction ---
if st.button("Analyser la commande"):
    with st.spinner('Analyse en cours...'):
        time.sleep(1) # Simulation de calcul
        
        # Création du DataFrame pour le modèle
        data = pd.DataFrame([[distance, montant]], columns=['distance_km', 'order_total'])
        
        # Prédiction
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0]
        
        # --- 7. Affichage des résultats ---
        st.markdown("---")
        if prediction == 1:
            st.error("### ⚠️ Statut Prévu : RETARDÉE (Delayed)")
            st.write(f"**Confiance du modèle :** {proba[1]*100:.2f}%")
        else:
            st.success("### ✅ Statut Prévu : À L'HEURE (On Time)")
            st.write(f"**Confiance du modèle :** {proba[0]*100:.2f}%")

# --- 8. Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Projet : Analyse Glovo Tunisie\nAlgorithme : Random Forest")