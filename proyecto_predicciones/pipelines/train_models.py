import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def cargar_datos():
    return pd.read_csv("data/Student_Performance_Procesado.csv")

def entrenar_modelos(X, y):
    modelos = {
        "linear_model.pkl": LinearRegression(),
        "svr_model.pkl": SVR(kernel="linear"),
        "decision_tree.pkl": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        with open(f"models/{nombre}", "wb") as archivo:
            pickle.dump(modelo, archivo)
        print(f"Modelo {nombre} guardado.")

if __name__ == "__main__":
    datos = cargar_datos()
    X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = datos['Performance Index']
    entrenar_modelos(X, y)
