# Imagen para desplegar API + interfaz juntas (ej. Hugging Face Spaces, Render, etc.)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces espera el puerto 7860; la API corre interna en el 8000.
ENV API_URL=http://127.0.0.1:8000
EXPOSE 7860

CMD ["bash", "start.sh"]
