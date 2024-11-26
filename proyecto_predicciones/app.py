import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Título principal del tablero
st.title("Tablero de Rendimiento Estudiantil")

# Breve explicación del tablero
st.write("""
Este tablero presenta un análisis de rendimiento estudiantil en función de varias características como las horas de estudio, puntajes previos, actividades extracurriculares, horas de sueño y cuestionarios practicados.
Utilizando distintos modelos de regresión, se predice un índice de rendimiento (Performance Index) basado en estas características.
""")

# Cargar datos procesados desde archivo
datos = pd.read_csv("data/Student_Performance_Procesado.csv")

# Cargar modelos entrenados
def cargar_modelos():
    modelos = {
        "Regresión Lineal": pickle.load(open("models/linear_model.pkl", "rb")),
        "SVR": pickle.load(open("models/svr_model.pkl", "rb")),
        "Árbol de Decisión": pickle.load(open("models/decision_tree.pkl", "rb"))
    }
    return modelos

modelos = cargar_modelos()

# Separar las características (X) y la etiqueta (y)
X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = datos['Performance Index']

# Mostrar un formulario en la barra lateral para predicción
with st.sidebar.form("prediction_form"):
    st.header("Seleccione el Modelo para la Predicción")
    seleccion_modelo = st.selectbox("Modelo", ["Regresión Lineal", "SVR", "Árbol de Decisión"])

    # Permitir el ingreso de parámetros para la predicción
    st.subheader("Ingrese los Parámetros para la Predicción")
    horas_estudio = st.number_input("Horas de Estudio (Min: 0, Max: 10)", min_value=0, max_value=10)
    puntaje_previo = st.number_input("Puntaje Previo (Min: 0, Max: 100)", min_value=0, max_value=100)
    horas_sueno = st.number_input("Horas de Sueño (Min: 0, Max: 12)", min_value=0, max_value=12)
    actividad_extra = st.selectbox("Actividades Extracurriculares", options=["No", "Sí"])
    cuestionarios_practicados = st.number_input("Cuestionarios Practicados (Min: 0, Max: 10)", min_value=0, max_value=10)
    valor_actividad_extra = 1 if actividad_extra == "Sí" else 0

    # Botón para realizar la predicción
    submitted = st.form_submit_button("Evaluar")
    if submitted:
        # Crear un DataFrame con los parámetros de entrada
        input_datos = pd.DataFrame([[horas_estudio, puntaje_previo, valor_actividad_extra, horas_sueno, cuestionarios_practicados]],
                                   columns=X.columns)

        # Realizar la predicción
        modelo_seleccionado = modelos[seleccion_modelo]
        prediccion = modelo_seleccionado.predict(input_datos)
        st.write(f"Predicción de Performance Index ({seleccion_modelo}): {prediccion[0]:.2f}")
