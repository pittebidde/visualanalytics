import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt 


# Set page configuration for full width
st.set_page_config(layout="wide")

# Title row - using columns to center the title


st.title("Parkinson detection - Vergleich von gesunden Probanden und Personen mit Parkinsons")
# Load all subject data
subject_data = {
    "CNNPT_AEPT_TPT": pd.read_csv("CNNCT_AECT_TCT.csv", index_col=0),
    "CNNCT_AECT_TCT": pd.read_csv("CNNPT_AEPT_TPT.csv", index_col=0),
    "CNNPT_AEPT_TCT": pd.read_csv("CNNPT_AEPT_TCT.csv", index_col=0),
    "CNNPT_AECT_TPT": pd.read_csv("CNNPT_AECT_TPT.csv", index_col=0)
}
subject_labels = {
    "CNNCT_AECT_TCT": "Beide Modelle sagen erfolgreich Parkinson voraus",
    "CNNPT_AEPT_TPT": "Beide Modelle sagen erfolgreich Control voraus",
    "CNNPT_AECT_TPT": "CNN sagt erfolgreich Parkinson voraus, Autoencoder sagt fälschlicherweise Control voraus",
    "CNNPT_AEPT_TCT": "Beide Modelle scheitern daran, einen Parkinsons Proband zu identifizieren"
}
AE_reconstrution = {
    "AECTTCT.csv": pd.read_csv("AECTTCT.csv", index_col=0),
    "AEPTTPT.csv": pd.read_csv("AEPTTPT.csv", index_col=0),
    "AEPTTCT.csv": pd.read_csv("AEPTTCT.csv", index_col=0),
    "AECTTPT.csv": pd.read_csv("AECTTPT.csv", index_col=0)
}



# Subject selection row
st.header("Probandenwahl")
selected_subject = st.selectbox(
    "Szenario:",
    options=list(subject_data.keys()),
    format_func=lambda x: subject_labels[x],  # This shows the friendly label
    index=0,
    key="subject_selector"
)

# Get data for selected subject
current_data = subject_data[selected_subject]

# Determine if control or parkinsons
subject_type_CNN = "Parkinsons" if "CNNCT_AECT_TCT" in selected_subject else "Control"
subject_type_AE = "Parkinsons" if ("CNNCT_AECT_TCT" in selected_subject) or ("CNNPT_AECT_TPT" in selected_subject) else "Control"
subject_type_True = "Parkinsons" if ("CNNCT_AECT_TCT" in selected_subject) or ("CNNPT_AEPT_TCT" in selected_subject) else "Control"



# Add some spacing
st.write("")

# First row with 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Autoencoder prediction")
    # Display either Control or Parkinsons with appropriate styling
    if subject_type_AE == "Control":
        st.markdown("""
        <div style='
            background-color: #CBC6B9;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        CONTROL
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background-color: #3D3F40;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        PARKINSONS
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.subheader("True prediction")
    # Display either Control or Parkinsons with appropriate styling
    if subject_type_True == "Control":
        st.markdown("""
        <div style='
            background-color: #CBC6B9;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        CONTROL
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background-color: #3D3F40;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        PARKINSONS
        </div>
        """, unsafe_allow_html=True)

    
with col3:
    st.subheader("CNN prediction")
    # Display either Control or Parkinsons with appropriate styling
    if subject_type_CNN == "Control":
        st.markdown("""
        <div style='
            background-color: #CBC6B9;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        CONTROL
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='
            background-color: #3D3F40;
            color: white;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        '>
        PARKINSONS
        </div>
        """, unsafe_allow_html=True)

# Second row with 3 columns
col4, col5, col6 = st.columns(3)

with col4:
    # Determine which AE reconstruction to show based on selected subject
    ae_mapping = {
        "CNNCT_AECT_TCT": "AECTTCT.csv",
        "CNNPT_AEPT_TPT": "AEPTTPT.csv",
        "CNNPT_AEPT_TCT": "AEPTTCT.csv",
        "CNNPT_AECT_TPT": "AECTTPT.csv"
    }

    selected_ae_file = ae_mapping[selected_subject]
    ae_data = AE_reconstrution[selected_ae_file]

    # Create a plot for the AE reconstruction data
    st.subheader("Autoencoder Reconstruction")
    
    # Get the two time series (assuming the AE data has exactly two rows)
    if len(ae_data) >= 2:
        # Transpose the data so time is on x-axis
        ae_data_transposed = ae_data.T
    
        # Create Plotly figure
        fig = go.Figure()
    
        # Add the two time series
        fig.add_trace(go.Scatter(
            x=ae_data_transposed.index,
            y=ae_data_transposed.iloc[:, 0],
            name="Originaldaten",
            line=dict(color='blue')
        ))
    
        fig.add_trace(go.Scatter(
            x=ae_data_transposed.index,
            y=ae_data_transposed.iloc[:, 1],
            name="Rekonstruktion",
            line=dict(color='green')
        ))
    
        # Fill the area between the two series
        fig.add_trace(go.Scatter(
            x=ae_data_transposed.index.tolist() + ae_data_transposed.index.tolist()[::-1],
            y=ae_data_transposed.iloc[:, 0].tolist() + ae_data_transposed.iloc[:, 1].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 182, 193, 0.5)',  # Light red with 50% opacity
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name="Difference"
        ))
    
        # Update layout
        fig.update_layout(
            height=500,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # Display the actual dataframe
    else:
        st.warning("Autoencoder reconstruction data doesn't contain enough rows for comparison")

########################
#Feature importance CNN#
########################
with col5:
    # Main visualization in the center
    st.subheader("Rohdatenvisualisierung")
    
    selected_sensors = st.multiselect(
        "Rohdaten der Sensoren:",
        options=current_data.columns.tolist(),
        default=current_data.columns[:3].tolist(),
        key="sensor_selector"
    )

    if selected_sensors:
        st.line_chart(
            current_data[selected_sensors],
            height=450,
            use_container_width=True
        )
    else:
        st.warning("Please select at least one sensor to plot")

########################
#Feature importance CNN#
########################
with col6:
    st.subheader("Feature Importance CNN")
    
    # Sample feature importance data - replace this with your actual data
    feature_importance = {
        "Feature 1": 0.0040,
        "Feature 2": 0.0000,
        "Feature 3": 0.0120,
        "Feature 4": -0.0200,
        "Feature 5": -0.0040,
        "Feature 6": -0.0200,
        "Feature 7": -0.000,
        "Feature 8": -0.0040,
        "Feature 9": 0.0120,
        "Feature 10": 0.0000,
        "Feature 11": -0.0040,
        "Feature 12": -0.0040,
        "Feature 13": 0.0040,
        "Feature 14": 0.0040,
        "Feature 15": -0.0040,
        "Feature 16": -0.0040,
        "Feature 17": -0.0080,
        "Feature 18": -0.0120
    }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort features by importance value (ascending)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1])
    features, importances = zip(*sorted_features)
    
    # Create color gradient
    negative_importances = [imp for imp in importances if imp < 0]
    if negative_importances:
        min_neg = min(negative_importances)
        max_neg = max(negative_importances)
    else:
        min_neg, max_neg = -1, 0

    colors = []
    for imp in importances:
        if imp > 0:
            intensity = 0.5 + (imp / (2 * max(abs(i) for i in importances if i != 0)))
            colors.append((0, intensity, 0, 0.7))
        elif imp < 0:
            if min_neg == max_neg:
                norm_imp = 0.5
            else:
                norm_imp = 0.5 + 0.5 * (imp - min_neg) / (max_neg - min_neg)
            colors.append((1, 0.8 * norm_imp, 0.8 * norm_imp, 0.7))
        else:
            colors.append((0.5, 0.5, 0.5, 0.7))

    # Create the bars
    bars = ax.barh(features, importances, color=colors)

    # Add value labels INSIDE the bars
    for bar in bars:
        width = bar.get_width()
        if width != 0:
            # Position text inside the bar
            x_pos = width/2 if width > 0 else width/2
            color = 'black' if abs(width) > 0.005 else 'black'  # White for prominent values
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', 
                    va='center', ha='center',
                    color=color, fontsize=9)

    ax.set_xlabel('Importance Score')
    ax.set_title('CNN Feature Importance Ranking')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Add compact legend
    positive_patch = plt.Rectangle((0,0),1,1,fc='green', alpha=0.7)
    negative_patch = plt.Rectangle((0,0),1,1,fc='red', alpha=0.7)
    zero_patch = plt.Rectangle((0,0),1,1,fc='gray', alpha=0.7)
    ax.legend([positive_patch, negative_patch, zero_patch], 
              ['+', '-', '0'],
              title='Impact',
              loc='lower right')

    # Remove spines for cleaner look
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    # Display in Streamlit
    st.pyplot(fig)


##########
#Markdown#
##########
st.markdown("""
    <style>
        .main > div {
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .stMultiSelect [data-baseweb=select] {
            min-height: 42px;
        }
        .stSelectbox [data-baseweb=select] {
            min-height: 42px;
        }
    </style>
""", unsafe_allow_html=True)


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





#
#
#
#
#
#
##
#


