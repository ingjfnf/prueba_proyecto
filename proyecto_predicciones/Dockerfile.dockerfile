FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

# Crear las carpetas necesarias
RUN mkdir -p data models

# Ejecutar el preprocesamiento y entrenamiento
RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

EXPOSE 8001

CMD ["streamlit", "run", "app.py", "--server.port=8001", "--server.address=0.0.0.0"]
