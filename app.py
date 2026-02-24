import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Galopp-KI 2026", layout="wide")
st.title("🐎 KI-Vorhersage: Galoppsport 2026")

# 1. Daten laden
@st.cache_resource
def load_data():
    modell = joblib.load('galopp_ki_modell_v1.pkl')
    pferde = pd.read_csv('pferde_datenbank_2025.csv')
    jockeys = pd.read_csv('jockey_datenbank_2025.csv')
    trainer = pd.read_csv('trainer_datenbank_2025.csv')
    return modell, pferde, jockeys, trainer

try:
    modell, df_pferde, df_jockeys, df_trainer = load_data()
    st.sidebar.success("✅ KI-Modell und Datenbanken sind aktiv!")
except Exception as e:
    st.error(f"Fehler beim Laden: {e}")

# 2. Upload
uploaded_file = st.file_uploader("Starterliste 2026 hochladen (CSV-Format)", type=['csv'])

if uploaded_file is not None:
    neue_rennen = pd.read_csv(uploaded_file)
    st.write("📋 Deine hochgeladenen Starter:")
    st.dataframe(neue_rennen[['horse_name', 'jockey', 'trainer', 'ml_quote']])
    
    if st.button("🔮 Vorhersage & Value berechnen"):
        with st.spinner('Gleiche historische Daten ab und berechne Wahrscheinlichkeiten...'):
            
            # --- DATEN ZUSAMMENFÜHREN ---
            # Wir suchen für jedes Pferd, Jockey und Trainer die alten Statistiken aus 2024/2025
            df = pd.merge(neue_rennen, df_pferde[['horse_name', 'bisherige_starts', 'bisherige_siege', 'siegrate_historisch', 'race_date']], on='horse_name', how='left')
            df = pd.merge(df, df_jockeys[['jockey', 'jockey_starts', 'jockey_siegrate']], on='jockey', how='left')
            df = pd.merge(df, df_trainer[['trainer', 'trainer_starts', 'trainer_siegrate']], on='trainer', how='left')
            
            # Fehlende Werte (Debütanten) mit 0 auffüllen
            fill_cols = ['bisherige_starts', 'bisherige_siege', 'siegrate_historisch', 'jockey_starts', 'jockey_siegrate', 'trainer_starts', 'trainer_siegrate']
            df[fill_cols] = df[fill_cols].fillna(0)
            
            # Fitness berechnen (Tage seit letztem Rennen)
            df['race_date_heute'] = pd.to_datetime(df['race_date_x'])
            df['race_date_alt'] = pd.to_datetime(df['race_date_y'])
            df['tage_seit_letztem_rennen'] = (df['race_date_heute'] - df['race_date_alt']).dt.days
            df['tage_seit_letztem_rennen'] = df['tage_seit_letztem_rennen'].fillna(30) # 30 Tage als Standard für Debütanten
            
            # Kategoriale Variablen umwandeln (Dummies)
            df_modell = pd.get_dummies(df, columns=['surface', 'gender', 'venue'])
            
            # WICHTIG: Das Modell erwartet exakt die gleichen Spalten wie beim Training. 
            # Wir holen uns die benötigten Spaltennamen aus dem Modell selbst!
            features_vom_modell = modell.estimator.feature_names_in_
            
            # Wir füllen Spalten, die in der neuen Datei fehlen (z.B. eine Rennbahn, die heute nicht dabei ist) mit 0
            for col in features_vom_modell:
                if col not in df_modell.columns:
                    df_modell[col] = 0
            
            # Wir sortieren die Spalten in die exakt richtige Reihenfolge
            X_live = df_modell[features_vom_modell]
            
            # --- VORHERSAGE ---
            wahrscheinlichkeiten = modell.predict_proba(X_live)[:, 1]
            df['KI_Sieg_Wahrscheinlichkeit'] = wahrscheinlichkeiten
            
            # --- VALUE BERECHNEN (SMART STAKING) ---
            df['Markt_Wahrscheinlichkeit'] = 1 / df['ml_quote']
            df['Edge (Vorsprung)'] = df['KI_Sieg_Wahrscheinlichkeit'] - df['Markt_Wahrscheinlichkeit']
            
            # Einsatzempfehlung: (Edge * 100) * 2 Euro (max 20 Euro)
            df['Empfohlener_Einsatz_€'] = (df['Edge (Vorsprung)'] * 100 * 2)
            df['Empfohlener_Einsatz_€'] = df['Empfohlener_Einsatz_€'].clip(lower=0, upper=20).round(2)
            
            # Nur gute Wetten anzeigen (Edge über 2%)
            df['Ist_Value_Bet'] = df['Edge (Vorsprung)'] > 0.02
            
            # --- ERGEBNIS ANZEIGEN ---
            st.success("Berechnung abgeschlossen!")
            
            # Schöne Tabelle für die Ausgabe formatieren
            ausgabe = df[['horse_name', 'jockey', 'ml_quote', 'KI_Sieg_Wahrscheinlichkeit', 'Edge (Vorsprung)', 'Empfohlener_Einsatz_€', 'Ist_Value_Bet']].copy()
            ausgabe['KI_Sieg_Wahrscheinlichkeit'] = (ausgabe['KI_Sieg_Wahrscheinlichkeit'] * 100).round(2).astype(str) + ' %'
            ausgabe['Edge (Vorsprung)'] = (ausgabe['Edge (Vorsprung)'] * 100).round(2).astype(str) + ' %'
            
            st.subheader("💡 Deine Empfehlungen für das Rennen")
            st.dataframe(ausgabe.sort_values(by='Empfohlener_Einsatz_€', ascending=False), use_container_width=True)
            
            st.info("Pferde mit einem Einsatz von 0 € bieten aktuell keinen Value gegenüber den Buchmachern.")
