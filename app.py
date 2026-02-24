import streamlit as st
import pandas as st_pd
import pandas as pd
import joblib

st.set_page_config(page_title="Galopp-KI 2026", layout="wide")

st.title("🐎 KI-Vorhersage: Galoppsport 2026")
st.markdown("Lade hier die Starterliste für das nächste Rennen hoch, um Value-Bets zu finden!")

# 1. Modell und historische Daten laden
@st.cache_resource
def load_data():
    modell = joblib.load('galopp_ki_modell_v1.pkl')
    pferde = pd.read_csv('pferde_datenbank_2025.csv')
    jockeys = pd.read_csv('jockey_datenbank_2025.csv')
    trainer = pd.read_csv('trainer_datenbank_2025.csv')
    return modell, pferde, jockeys, trainer

try:
    modell, df_pferde, df_jockeys, df_trainer = load_data()
    st.success("✅ Modell und Datenbanken erfolgreich geladen!")
except Exception as e:
    st.error(f"Fehler beim Laden der Dateien: {e}")

# 2. Upload der neuen Starterliste
uploaded_file = st.file_uploader("Starterliste hochladen (CSV)", type=['csv'])

if uploaded_file is not None:
    neue_rennen = pd.read_csv(uploaded_file)
    st.write("Vorschau der hochgeladenen Starterliste:")
    st.dataframe(neue_rennen.head())
    
    if st.button("🔮 Vorhersage & Value berechnen"):
        st.info("Daten werden mit Historie abgeglichen und Wahrscheinlichkeiten berechnet... (Funktion in Kürze verfügbar)")
        # Hier kommt im nächsten Schritt unsere Zusammenführung rein!
