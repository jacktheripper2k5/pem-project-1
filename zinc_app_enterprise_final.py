import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.markdown("""
<style>
body {background-color: #242424; color: white;}
section[data-testid="stSidebar"] {
    background-color: #1A1A1A;
    border-right: 1px solid #444;
}
h1, h2, h3 {color: white;}
</style>
""", unsafe_allow_html=True)

st.title("Zinc Recovery Optimization from Blast Furnace Slag")

@st.cache_data
def load_data():
    return pd.read_csv("zinc_recovery_dataset.csv")

df = load_data()

X = df.drop("Zinc Recovery Efficiency (%)", axis=1)
y = df["Zinc Recovery Efficiency (%)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

st.sidebar.header("Adjust Process Parameters")

temperature = st.sidebar.slider("Slag Temperature (°C)", 1150, 1300, 1225)
coal_air = st.sidebar.slider("Coal-to-Air Ratio", 0.6, 1.4, 1.0)
basicity = st.sidebar.slider("Slag Basicity (CaO/SiO2)", 0.7, 1.5, 1.1)
zn_initial = st.sidebar.slider("Initial Zinc (%)", 2.0, 12.0, 6.0)
feo = st.sidebar.slider("FeO Content (%)", 5.0, 25.0, 12.0)

input_data = pd.DataFrame({
    "Slag Temperature (°C)": [temperature],
    "Coal-to-Air Ratio": [coal_air],
    "Slag Basicity (CaO/SiO2)": [basicity],
    "Initial Zinc in Slag (%)": [zn_initial],
    "FeO content in Slag (%)": [feo]
})

prediction = model.predict(input_data)[0]

st.subheader("Predicted Zinc Recovery Efficiency")
st.markdown(f"<h1 style='color:#FFD700'>{prediction:.2f}%</h1>", unsafe_allow_html=True)

st.subheader("Feature Importance")

importance = model.feature_importances_
features = X.columns

colors = ["#E91E63", "#9C27B0", "#00BCD4", "#FF9800", "#8BC34A"]

fig, ax = plt.subplots()
ax.barh(features, importance, color=colors)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

for label in ax.get_yticklabels():
    label.set_color("black")

for label in ax.get_xticklabels():
    label.set_color("black")

st.pyplot(fig)

st.subheader("Sensitivity Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    temps = np.linspace(1150, 1300, 50)
    preds = []
    for t in temps:
        test = input_data.copy()
        test["Slag Temperature (°C)"] = t
        preds.append(model.predict(test)[0])
    fig, ax = plt.subplots()
    ax.plot(temps, preds, color="#E91E63")
    ax.set_facecolor("#242424")
    fig.patch.set_facecolor("#242424")
    ax.tick_params(colors='white')
    ax.set_title("Temperature Effect", color="white")
    st.pyplot(fig)

with col2:
    vals = np.linspace(0.7, 1.5, 50)
    preds = []
    for b in vals:
        test = input_data.copy()
        test["Slag Basicity (CaO/SiO2)"] = b
        preds.append(model.predict(test)[0])
    fig, ax = plt.subplots()
    ax.plot(vals, preds, color="#00BCD4")
    ax.set_facecolor("#242424")
    fig.patch.set_facecolor("#242424")
    ax.tick_params(colors='white')
    ax.set_title("Basicity Effect", color="white")
    st.pyplot(fig)

with col3:
    vals = np.linspace(0.6, 1.4, 50)
    preds = []
    for c in vals:
        test = input_data.copy()
        test["Coal-to-Air Ratio"] = c
        preds.append(model.predict(test)[0])
    fig, ax = plt.subplots()
    ax.plot(vals, preds, color="#9C27B0")
    ax.set_facecolor("#242424")
    fig.patch.set_facecolor("#242424")
    ax.tick_params(colors='white')
    ax.set_title("Coal-Air Effect", color="white")
    st.pyplot(fig)
