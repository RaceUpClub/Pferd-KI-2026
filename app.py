import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. PAGE CONFIG & TERMINAL DESIGN ---
st.set_page_config(page_title="Galopp-KI 2026", layout="wide")

# CSS für den Hacker/Terminal-Look injizieren
st.markdown("""
    <style>
    /* Haupt-Hintergrund und Textfarbe */
    .stApp {
        background-color: #0d1117;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Überschriften */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #00ff41 !important;
        font-family: 'Courier New', Courier, monospace !important;
    }

    /* Der Button */
    div.stButton > button:first-child {
        background-color: #000000;
        color: #00ff41;
        border: 2px solid #00ff41;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #00ff41;
        color: #000000;
        border: 2px solid #00ff41;
    }
    
    /* Info-Boxen und Warnungen anpassen */
    .stAlert {
        background-color: #000000 !important;
        border: 1px solid #00ff41 !important;
        color: #00ff41 !important;
    }
    
    /* Dateiupload-Bereich */
    .stFileUploader > div > div {
        background-color: #000000;
        border: 1px dashed #00ff41;
    }
    </style>
""", unsafe_allow_html=True)


st.title("/> Galopp-KI_2026.exe")
st.markdown("`[SYSTEM BEREIT] Initialisiere Wett-Algorithmus... Bitte Starterliste (.csv) einspeisen.`")

# --- 2. DATEN LADEN ---
@st.cache_resource
def load_data():
    modell = joblib.load('galopp_ki_modell_v2.pkl')
    pferde = pd.read_csv('pferde_datenbank_2026_ml_ready.csv')
    jockeys = pd.read_csv('jockey_datenbank_master.csv')
    trainer = pd.read_csv('trainer_datenbank_master.csv')
    return modell, pferde, jockeys, trainer

try:
    modell, df_pferde, df_jockeys, df_trainer = load_data()
    st.success("> DATENBANKEN GELADEN. VERBINDUNG HERGESTELLT.")
except Exception as e:
    st.error(f"FATAL ERROR: {e}")

# --- 3. UPLOAD & VERARBEITUNG ---
uploaded_file = st.file_uploader("", type=['csv'])

if uploaded_file is not None:
    neue_rennen = pd.read_csv(uploaded_file)
    st.markdown("`> ANALYSIERE STARTERLISTE...`")
    st.dataframe(neue_rennen[['horse_name', 'jockey', 'trainer', 'ml_quote']])
    
    if st.button("> EXECUTE: Vorhersage & Value berechnen"):
        with st.spinner('Kalkuliere Wahrscheinlichkeiten...'):
            
            # 🚨 DER FIX: Logbuch nach Datum sortieren und Klone löschen!
            df_pferde['race_date'] = pd.to_datetime(df_pferde['race_date'])
            df_pferde_unique = df_pferde.sort_values(by='race_date').drop_duplicates(subset=['horse_name'], keep='last')
            
            df = pd.merge(neue_rennen, df_pferde_unique[['horse_name', 'bisherige_starts', 'bisherige_siege', 'siegrate_historisch']], on='horse_name', how='left')
            df = pd.merge(df, df_jockeys[['jockey', 'jockey_starts', 'jockey_siegrate']], on='jockey', how='left')
            df = pd.merge(df, df_trainer[['trainer', 'trainer_starts', 'trainer_siegrate']], on='trainer', how='left')
            
            fill_cols = ['bisherige_starts', 'bisherige_siege', 'siegrate_historisch', 'jockey_starts', 'jockey_siegrate', 'trainer_starts', 'trainer_siegrate']
            df[fill_cols] = df[fill_cols].fillna(0)
            
            df['race_date_heute'] = pd.to_datetime(df['race_date_x'])
            df['race_date_alt'] = pd.to_datetime(df['race_date_y'])
            df['tage_seit_letztem_rennen'] = (df['race_date_heute'] - df['race_date_alt']).dt.days
            df['tage_seit_letztem_rennen'] = df['tage_seit_letztem_rennen'].fillna(30)
            
            df_modell = pd.get_dummies(df, columns=['surface', 'gender', 'venue'])
            if hasattr(modell, 'estimator'):
                features_vom_modell = modell.estimator.feature_names_in_
            else:
                features_vom_modell = modell.feature_names_in_
            
            for col in features_vom_modell:
                if col not in df_modell.columns:
                    df_modell[col] = 0
            
            X_live = df_modell[features_vom_modell]
            
            # --- VORHERSAGE ---
            wahrscheinlichkeiten = modell.predict_proba(X_live)[:, 1]
            df['KI_Sieg_Wahrscheinlichkeit'] = wahrscheinlichkeiten
            
            # --- VALUE BERECHNEN ---
            df['Markt_Wahrscheinlichkeit'] = 1 / df['ml_quote']
            df['Edge'] = df['KI_Sieg_Wahrscheinlichkeit'] - df['Markt_Wahrscheinlichkeit']
            
            df['Einsatz_€'] = (df['Edge'] * 100 * 2).clip(lower=0, upper=20).round(2)
            df['Value_Bet'] = df['Edge'] > 0.02
            
            st.success("> KALKULATION ABGESCHLOSSEN.")
            
            ausgabe = df[['horse_name', 'jockey', 'ml_quote', 'KI_Sieg_Wahrscheinlichkeit', 'Edge', 'Einsatz_€', 'Value_Bet']].copy()
            ausgabe['KI_Sieg_Wahrscheinlichkeit'] = (ausgabe['KI_Sieg_Wahrscheinlichkeit'] * 100).round(2).astype(str) + ' %'
            ausgabe['Edge'] = (ausgabe['Edge'] * 100).round(2).astype(str) + ' %'
            
            st.subheader("/> OUTPUT: WETT-EMPFEHLUNGEN")
            st.dataframe(ausgabe.sort_values(by='Einsatz_€', ascending=False), use_container_width=True)
