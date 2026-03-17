import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# -----------------------------
# CARGAR MODELOS
# -----------------------------
modelo = tf.keras.models.load_model("mejor_modelo.keras", compile=False)
scaler = joblib.load("minmax_scaler.joblib")
pca = joblib.load("modelo_pca.joblib")
encoders = joblib.load("label_encoders.joblib")

le_payment = encoders["Payment_of_Min_Amount"]

# -----------------------------
# INTERFAZ
# -----------------------------
st.title("Predicción de Credit Score")

st.image(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQAfZXKT8-ix4vRN-n4KqDKHsrhUnxhgUCCIQ&s",
    caption="Análisis de crédito",
    use_container_width=True
)

st.write("Ingrese la información financiera para predecir el **Credit Score**.")

# -----------------------------
# INPUTS
# -----------------------------

num_bank_accounts = st.slider(
    "Número de cuentas bancarias",
    0, 10, 5
)

num_credit_card = st.slider(
    "Número de tarjetas de crédito",
    1, 10, 5
)

interest_rate = st.slider(
    "Tasa de interés",
    1.0, 34.0, 10.0
)

delay_due = st.slider(
    "Retraso desde la fecha de pago(Dias)",
    2, 63, 10
)

num_delayed = st.slider(
    "Número de pagos retrasados",
    0, 26, 0
)

num_inquiries = st.slider(
    "Número de consultas de crédito",
    0.0, 16.375, 0.0
)

outstanding_debt = st.slider(
    "Deuda pendiente",
    1.0, 4998.00, 100.0
)

payment_min = st.selectbox(
    "Pago del monto mínimo",
    le_payment.classes_
)

# -----------------------------
# PREDICCION
# -----------------------------

if st.button("Predecir Puntuación Crediticia"):

    # aplicar label encoder
    payment_encoded = le_payment.transform([payment_min])[0]

    datos = np.array([[
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        delay_due,
        num_delayed,
        num_inquiries,
        outstanding_debt,
        payment_encoded
    ]])

    # PCA
    datos_pca = pca.transform(datos)

    # Escalar
    datos_scaled = scaler.transform(datos_pca)

    # Predicción
    pred = modelo.predict(datos_scaled)
    clase = np.argmax(pred)

    clases = ["Mala", "Normal", "Buena"]
    resultado = clases[clase]

    st.subheader("Resultado de la Predicción")

    if resultado == "Mala":
        st.error(f"Predicción de Puntuación Crediticia: **{resultado}**")

    elif resultado == "Normal":
        st.info(f"Predicción de Puntuación Crediticia: **{resultado}**")

    else:
        st.success(f"Predicción de Puntuación Crediticia: **{resultado}**")


