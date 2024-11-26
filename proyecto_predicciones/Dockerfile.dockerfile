FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install -r requirements.txt

RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

EXPOSE 8001

CMD ["streamlit", "run", "app.py", "--server.port=8001", "--server.address=0.0.0.0"]
