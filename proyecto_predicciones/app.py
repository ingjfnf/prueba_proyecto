import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Establecemos el título principal del tablero
st.title("Tablero de Rendimiento Estudiantil")

# Proporcionamos una breve explicación del tablero
st.write("""
Este tablero presenta un análisis de rendimiento estudiantil en función de varias características como las horas de estudio, puntajes previos, actividades extracurriculares, horas de sueño y cuestionarios practicados.
Utilizando distintos modelos de regresión, se predice un índice de rendimiento (Performance Index) basado en estas características.
""")

# Cargamos los datos procesados desde la carpeta 'data'
datos = pd.read_csv("data/Student_Performance_Procesado.csv")

# Seleccionamos las columnas especificadas para la predicción
X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = datos['Performance Index']

# Cargamos los modelos entrenados desde la carpeta 'models'
modelos = {
    "Regresión Lineal": pickle.load(open("models/linear_model.pkl", "rb")),
    "SVR": pickle.load(open("models/svr_model.pkl", "rb")),
    "Árbol de Decisión": pickle.load(open("models/decision_tree.pkl", "rb"))
}

# Calculamos las métricas de cada modelo
metricas = []
for nombre, modelo in modelos.items():
    y_predicho = modelo.predict(X)
    mse = mean_squared_error(y, y_predicho)
    mae = mean_absolute_error(y, y_predicho)
    r2 = r2_score(y, y_predicho)
    metricas.append({"Modelo": nombre, "MSE": mse, "MAE": mae, "R²": r2})

# Presentamos la tabla de comparación de métricas en el centro
with st.columns([1, 2, 1])[1]:
    st.subheader("Comparación de Métricas de Modelos")
    st.dataframe(pd.DataFrame(metricas))

# Identificamos el mejor modelo basado en el MSE y el R²
mejor_modelo_info = pd.DataFrame(metricas).sort_values(by=["MSE", "R²"], ascending=[True, False]).iloc[0]
nombre_mejor_modelo = mejor_modelo_info["Modelo"]

# Mostramos la sección "Mejor Modelo"
st.subheader("Mejor Modelo")
st.write(f"El mejor modelo es: **{nombre_mejor_modelo}**")
st.write("Este modelo se selecciona como el mejor porque tiene el menor **MSE** (Error Cuadrático Medio) "
         "y el mayor **R²** (Coeficiente de Determinación), lo cual indica una mejor capacidad de predicción "
         "y una mayor explicación de la variabilidad en los datos.")

# Graficamos Predicciones vs Valores Reales y el Histograma de Residuales para el mejor modelo
modelo_mejor = modelos[nombre_mejor_modelo]
y_predicho_mejor = modelo_mejor.predict(X)
residuales_mejor = y - y_predicho_mejor

st.subheader(f"Comportamiento del Mejor Modelo: {nombre_mejor_modelo}")
col1, col2 = st.columns(2)
with col2:
    st.write("**Predicciones vs. Valores Reales**")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(y, y_predicho_mejor, color="green", alpha=0.5, label="Predicciones (Puntos)")
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], linestyle='--', lw=2, color="red", label="Línea Ideal (Y=X)")
    ax1.set_xlabel("Valores Reales")
    ax1.set_ylabel("Predicciones")
    ax1.set_title(f"Predicciones vs. Valores Reales para {nombre_mejor_modelo}")
    ax1.legend()
    st.pyplot(fig1)
    st.markdown("""
    **Explicación:**  
    Esta gráfica muestra cómo las predicciones del modelo se comparan con los valores reales.  
    - Los puntos cercanos a la línea roja representan predicciones precisas.  
    - Si los puntos siguen un patrón alrededor de la línea, esto indica un buen ajuste del modelo.  

    **Análisis del resultado:**  
    - Los puntos están distribuidos muy cerca de la línea ideal, lo que indica un alto poder predictivo del modelo.  
    - No se observan patrones o desviaciones sistemáticas, lo que sugiere que el modelo captura correctamente la relación entre las variables.  
    """)

with col1:
    st.write("**Histograma de Residuales**")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.hist(residuales_mejor, bins=20, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Residuales")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de los Residuales")
    st.pyplot(fig2)
    st.markdown("""
    **Explicación:**  
    El histograma muestra la distribución de los residuales (diferencias entre los valores reales y las predicciones).  
    - Una distribución simétrica y centrada en 0 indica que el modelo no presenta sesgos significativos.  
    - Si aparecen valores extremos, podrían ser outliers o errores del modelo en capturar algunos patrones.  

    **Análisis del resultado:**  
    - La distribución simétrica y centrada en cero valida que el modelo no tiene sesgos significativos y que sus errores son aleatorios.  
    - No se observan valores extremos destacados, lo que sugiere que el modelo es robusto frente a outliers y maneja los datos adecuadamente.  
    """)

# Explicamos cada modelo utilizado
descripcion_modelos = {
    "Regresión Lineal": "La regresión lineal es un modelo simple que asume una relación lineal entre las variables independientes y la variable dependiente. Es útil cuando hay una tendencia directa y continua entre las variables.",
    "SVR": "El modelo SVR (Support Vector Regression) busca encontrar un límite de margen máximo que se ajuste a los datos dentro de un cierto rango de error. Es útil para datos con relaciones no lineales.",
    "Árbol de Decisión": "El árbol de decisión es un modelo que divide los datos en ramas según ciertas condiciones. Es adecuado cuando los datos tienen relaciones complejas y no lineales, y puede capturar interacciones entre variables."
}

# Creamos un formulario en la barra lateral para la predicción
with st.sidebar.form("prediction_form"):
    st.header("Seleccione el Modelo para la Predicción")
    seleccion_modelo = st.selectbox("Modelo", ["Regresión Lineal", "SVR", "Árbol de Decisión"])

    # Permitimos el ingreso de los parámetros para la predicción
    st.subheader("Ingrese los Parámetros para la Predicción")
    horas_estudio = st.number_input("Horas de Estudio (Min: 0, Max: 10)", min_value=0, max_value=10)
    puntaje_previo = st.number_input("Puntaje Previo (Min: 0, Max: 100)", min_value=0, max_value=100)
    horas_sueno = st.number_input("Horas de Sueño (Min: 0, Max: 12)", min_value=0, max_value=12)
    actividad_extra = st.selectbox("Actividades Extracurriculares", options=["No", "Sí"])
    cuestionarios_practicados = st.number_input("Cuestionarios Practicados (Min: 0, Max: 10)", min_value=0, max_value=10)
    valor_actividad_extra = 1 if actividad_extra == "Sí" else 0

    # Activamos el botón para calcular la predicción dentro del formulario
    submitted = st.form_submit_button("Evaluar")
    if submitted:
        # Creamos un DataFrame con los parámetros de entrada para la predicción
        input_datos = pd.DataFrame([[horas_estudio, puntaje_previo, valor_actividad_extra, horas_sueno, cuestionarios_practicados]],
                                   columns=['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced'])

        # Realizamos la predicción con el modelo seleccionado por el usuario
        modelo_seleccionado = modelos[seleccion_modelo]
        prediccion = modelo_seleccionado.predict(input_datos)
        st.write(f"Predicción de Performance Index ({seleccion_modelo}): {prediccion[0]:.2f}")

        # Mostramos la explicación del modelo seleccionado en la sección principal
        st.subheader("Explicación del Modelo Seleccionado")
        st.write(descripcion_modelos[seleccion_modelo])
