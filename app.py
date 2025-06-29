import os
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Parkinson Dashboard", layout="wide")


# Hilfsfunktion weil Bilder laden

def safe_image(path, caption=None, width=None, use_container_width=False):
    if os.path.exists(path):
        st.image(path, caption=caption, width=width, use_container_width=use_container_width)
    else:
        st.warning(f"Bild nicht gefunden: `{path}`")

# menü sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Start", "Dateninfo", "Gangvergleich", "AutoEncoder","Kontakt"],
        icons=[
            "house",        # Start
            "file-earmark-text",  # Dateninfo
            "bar-chart-line",     # Gangvergleich
            "activity",     # AutoEncoder
            "envelope",      # Kontakt

        ],
    )

# Startseite
if selected == "Start":
    
    # Auswahl Probanden
    probanden = {
        "Control 01": {
            "ae": "images/CT-CT.png",
            "gt": "ControlGroup",
            "gt_img": "images/CT.png",
            "rocket": "images/rocket.png",
            "ae_result": "ControlGroup",
            "rocket_result": "ControlGroup"
        },
        "Control 02": {
            "ae": "images/PT-CT.png",
            "gt": "ControlGroup",
            "gt_img": "images/CT-2.png",
            "rocket": "images/rocket.png",
            "ae_result": "Parkinson",
            "rocket_result": "Parkinson"
        },
        "Patient 01": {
            "ae": "images/PT-PT.png",
            "gt": "Parkinson",
            "gt_img":"images/PT.png",
            "rocket": "images/rocket.png",
            "ae_result": "Parkinson",
            "rocket_result": "Parkinson"
        },
        "Patient 02": {
            "ae": "images/CT-PT.png",
            "gt": "Parkinson",
            "gt_img": "images/PT-2.png",
            "rocket": "images/rocket.png",
            "ae_result": "ControlGroup",
            "rocket_result": "Parkinson"
        }
    }

    # Dropdown
    selected_proband = st.selectbox("Probanden auswählen", list(probanden.keys()))
    daten = probanden[selected_proband]

    # Erste Zeile: Textfelder oben
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.markdown(f"**Prediction AutoEncoder**  \n{daten['ae_result']}")
    with row1_col2:
        st.markdown(f"**Ground Truth**  \n{daten['gt']}")
    with row1_col3:
        st.markdown(f"**Prediction Rocket**  \n{daten['rocket_result']}")

    # Zweite Zeile: Bilder unten
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.image(daten["ae"], use_container_width=True)
    with row2_col2:
        st.image(daten["gt_img"], use_container_width=True)
    with row2_col3:
        st.image(daten["rocket"], use_container_width=True)




elif selected == "Dateninfo":
    st.title(" Informationen zum Datensatz")
    st.markdown("""

Dieses Dashboard basiert auf einer Ganganalyse-Datenbank mit Messungen von 93 Parkinson-Patient*innen und 73 gesunden Kontrollpersonen. Ziel der Studie war es, Gangmuster und deren Variabilität bei Morbus Parkinson zu untersuchen – insbesondere unter normalen Bedingungen und während einer kognitiven Doppelaufgabe.

Während des Gehens auf ebener Fläche wurden mithilfe von 16 Kraftsensoren (8 pro Fuß) Bodenreaktionskräfte aufgezeichnet (100 Hz). Die Daten ermöglichen die Analyse von Schrittdynamik, zeitlichen Merkmalen wie Schritt- oder Schwungzeit sowie der Gangvariabilität. Zusätzlich liegen demografische Daten und Informationen zum Krankheitsverlauf vor.
""")

    col1, col2 = st.columns([1, 2])  # Links 1/3, rechts 2/3

    with col1:
       safe_image("images/sensor_positionen.png", caption="Sensorpositionen", use_container_width=True)



    with col2:
        st.markdown("""
        ### Sensoranordnung an den Füßen

        Die Darstellung zeigt die Position der 8 Sensoren pro Fuß.  
        Sie ermöglichen eine präzise Erfassung der vertikalen Bodenreaktionskräfte während des Gehens.

        Die Sensoren liefern Daten zur Druckverteilung und erlauben Rückschlüsse auf Gangmuster, Stabilität und Schrittdynamik.
        """)

# Raw Data
elif selected == "Gangvergleich":
    st.title(" Vergleich: Kontrollperson vs. Parkinson-Patient*in")
    st.markdown("""
    In diesem Abschnitt wird ein direkter Vergleich von Gangmustern zwischen einer gesunden Kontrollperson und einer Parkinson-Patient*in dargestellt.
    """)

    @st.cache_data
    def load_data(filepath):
        df = pd.read_csv(filepath, delim_whitespace=True, header=None)
        time = df[0]
        sensors = df.iloc[:, 1:9]
        sensors.columns = [f"L{i}" for i in range(1, 9)]
        sensors["time"] = time
        return sensors

    # Dateien anpassen falls nötig
    control_file = "gait-in-parkinsons-disease-1.0.0/GaCo01_01.txt"
    patient_file = "gait-in-parkinsons-disease-1.0.0/GaPt03_01.txt"

    df_co = load_data(control_file)
    df_pt = load_data(patient_file)

    st.markdown("### Sensoren auswählen (linker Fuß):")

    # Sensor-Auswahl
    sensor_cols = [f"L{i}" for i in range(1,17)]
    select_all = st.checkbox("Alle Sensoren auswählen", value=True)

    selected_sensors = []
    cols = st.columns(8)
    for i, col in enumerate(cols):
        default = True if select_all else False
        if col.checkbox(sensor_cols[i], value=default, key=f"sensor_{i}"):
            selected_sensors.append(sensor_cols[i])

    if not selected_sensors:
        st.warning("Bitte mindestens einen Sensor auswählen.")
        st.stop()

    # Layout: Zwei nebeneinander
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kontrollperson")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        df_co.set_index("time")[selected_sensors].plot(ax=ax1)
        ax1.set_title("Gangprofil – Kontrolle")
        ax1.set_xlabel("Zeit (s)")
        ax1.set_ylabel("Kraft (N)")
        ax1.grid(True)
        ax1.legend(fontsize=7)
        st.pyplot(fig1)

    with col2:
        st.subheader("Parkinson-Patient*in")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_pt.set_index("time")[selected_sensors].plot(ax=ax2)
        ax2.set_title("Gangprofil – Patient*in mit Parkinson")
        ax2.set_xlabel("Zeit (s)")
        ax2.set_ylabel("Kraft (N)")
        ax2.grid(True)
        ax2.legend(fontsize=7)
        st.pyplot(fig2)


#autoencoder page 
elif selected == "AutoEncoder":
    st.title("AutoEncoder: Klassifikation im Vergleich")

    col1, col2 = st.columns(2)
    col1.markdown("### Control")
    col2.markdown("### Parkinsons")

    # Erste Zeile (True Label: Control)
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("**True: Control / Predicted: Control**")
        st.image("images/CT-CT.png", use_container_width=True)


    with row1_col2:
        st.markdown("**True: Control / Predicted: Parkinson**")
        st.image("images/PT-CT.png", use_container_width=True)


    # Zweite Zeile (True Label: Parkinson)
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("**True: Parkinson / Predicted: Control**")
        st.image("images/CT-PT.png", use_container_width=True)


    with row2_col2:
        st.markdown("**True: Parkinson / Predicted: Parkinson**")
        st.image("images/PT-PT.png", use_container_width=True)






#Closing Page mit mehr infos und kontakt
elif selected == "Kontakt":
    st.title("Generelle Informationen")
    st.markdown("""
                 Dieses Dashboard ist Teil der Proejktarbeit für das Modul "Visual Analytics". Ziel war es, mit XAI Methoden herauszufinden, welche Sensoren entscheidend für eine Klassifizierung des Probanden sind. 
               Der Original-Datensatz kann [hier](https://physionet.org/content/gaitpdb/1.0.0/) abgerufen werden. 
""")
    st.title("Kontakt")
    st.markdown("""
        Bei weiteren Fragen, können Sie sich gerne an die Autoren dieses Projektes wenden: 

        Philip Esswein 
         philip.esswein@studmail.htw-aalen.de

         Isabell Krüger
         isabell.krueger@studmail.htw-aalen.de
""")
