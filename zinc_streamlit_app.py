# ================================
# IMPORT LIBRARIES
# ================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ================================
# APP TITLE
# ================================
st.title("Zinc Recovery Optimization from Blast Furnace Slag")
st.markdown("""
This interactive tool simulates the **slag fuming process** used in industry 
to recover zinc from lead blast furnace slag.

The model is trained on synthetic data that incorporates real metallurgical behavior:
- Slag viscosity effects (Basicity)
- Magnetite formation (FeO + Temperature)
- Combustion efficiency (Coal/Air ratio)

Adjust the parameters in the sidebar to observe how recovery changes.
""")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("zinc_recovery_dataset.csv")

df = load_data()

# ================================
# PREPARE DATA FOR ML
# ================================
X = df.drop("Zinc Recovery Efficiency (%)", axis=1)
y = df["Zinc Recovery Efficiency (%)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# ================================
# SIDEBAR INPUTS
# ================================
st.sidebar.header("Adjust Process Parameters")

temperature = st.sidebar.slider(
    "Slag Temperature (°C)",
    1150, 1300, 1225
)

coal_air = st.sidebar.slider(
    "Coal-to-Air Ratio",
    0.6, 1.4, 1.0
)

basicity = st.sidebar.slider(
    "Slag Basicity (CaO/SiO₂)",
    0.7, 1.5, 1.1
)

zn_initial = st.sidebar.slider(
    "Initial Zinc (%)",
    2.0, 12.0, 6.0
)

feo = st.sidebar.slider(
    "FeO Content (%)",
    5.0, 25.0, 12.0
)

# ================================
# MAKE PREDICTION
# ================================
input_data = pd.DataFrame({
    "Slag Temperature (°C)": [temperature],
    "Coal-to-Air Ratio": [coal_air],
    "Slag Basicity (CaO/SiO2)": [basicity],
    "Initial Zinc in Slag (%)": [zn_initial],
    "FeO content in Slag (%)": [feo]
})

prediction = model.predict(input_data)[0]

# ================================
# DISPLAY RESULT
# ================================
st.subheader("Predicted Zinc Recovery Efficiency")

st.metric(
    label="Recovery (%)",
    value=f"{prediction:.2f}"
)

# ================================
# METALLURGICAL INSIGHT SECTION
# ================================
st.markdown("""
### Metallurgical Interpretation

- **Temperature ↑** → Enhances volatilization of Zn → Higher recovery  
- **Basicity too low** → Slag becomes viscous → Poor gas-solid interaction  
- **FeO high + low temperature** → Magnetite formation → Zinc gets trapped  
- **Coal/Air ratio imbalance** → Poor reduction conditions  

This model captures these nonlinear industrial trade-offs.
""")

# ================================
# FEATURE IMPORTANCE
# ================================
st.subheader("Feature Importance (Model Insight)")

importance = model.feature_importances_
features = X.columns

# Plot
fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_xlabel("Importance Score")
ax.set_title("Which Variables Affect Zinc Recovery Most?")

st.pyplot(fig)

# ================================
# RAW DATA VIEW (OPTIONAL)
# ================================
if st.checkbox("Show Raw Dataset"):
    st.write(df.head())
