import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Zinc Recovery Predictor", layout="wide")

st.title("Zinc Recovery Optimization from Blast Furnace Slag")

# Load data
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

# Sidebar inputs
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
st.metric(label="Recovery (%)", value=f"{prediction:.2f}")

# Interpretation
st.subheader("Metallurgical Interpretation")

if temperature > 1250:
    st.success("High temperature → better Zn volatilization")
elif temperature < 1180:
    st.warning("Low temperature → poor Zn recovery")
else:
    st.info("Moderate temperature")

if 0.9 <= coal_air <= 1.1:
    st.success("Optimal coal-air ratio")
else:
    st.warning("Coal-air imbalance")

if basicity < 0.9:
    st.error("Low basicity → high viscosity")
elif basicity > 1.3:
    st.warning("High basicity issues")
else:
    st.success("Optimal basicity")

if feo > 18 and temperature < 1200:
    st.error("Magnetite formation risk")
elif feo > 18:
    st.warning("High FeO")
else:
    st.success("FeO acceptable")

# Optimization
st.subheader("Suggested Optimal Conditions")

if st.button("Find Better Conditions"):
    best_pred = prediction
    best_params = (temperature, coal_air, basicity)

    for t in range(int(temperature-50), int(temperature+50), 25):
        for b in np.linspace(0.9, 1.3, 5):
            for c in np.linspace(0.9, 1.1, 5):
                test = pd.DataFrame({
                    "Slag Temperature (°C)": [t],
                    "Coal-to-Air Ratio": [c],
                    "Slag Basicity (CaO/SiO2)": [b],
                    "Initial Zinc in Slag (%)": [zn_initial],
                    "FeO content in Slag (%)": [feo]
                })
                pred = model.predict(test)[0]
                if pred > best_pred:
                    best_pred = pred
                    best_params = (t, c, b)

    st.success(f"Improved Recovery: {best_pred:.2f}%")
    st.write(f"Temperature: {best_params[0]} °C")
    st.write(f"Coal-Air Ratio: {best_params[1]:.2f}")
    st.write(f"Basicity: {best_params[2]:.2f}")

# Feature importance
st.subheader("Feature Importance")
importance = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importance)
st.pyplot(fig)
