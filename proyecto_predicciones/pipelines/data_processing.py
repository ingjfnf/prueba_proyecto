import pandas as pd

def cargar_datos():
    url = "https://raw.githubusercontent.com/Fibovin/des_modelos_1/refs/heads/main/Student_Performance.csv"
    return pd.read_csv(url)

def procesar_datos(datos):
    datos['Extracurricular Activities'] = datos['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return datos

def guardar_datos(datos):
    ruta = "data/Student_Performance_Procesado.csv"
    datos.to_csv(ruta, index=False)
    print(f"Datos procesados guardados en {ruta}")

if __name__ == "__main__":
    datos = cargar_datos()
    datos_procesados = procesar_datos(datos)
    guardar_datos(datos_procesados)
