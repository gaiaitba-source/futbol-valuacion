# Entregable 4 — Despliegue: API + Interfaz

Despliegue local de la solución del Entregable 3: una **API REST** que expone el modelo de
valuación de futbolistas y una **interfaz** (Streamlit) que la consume.

**Caso de uso:** un scout busca un jugador y obtiene su **valor estimado a 12 meses con un
rango de confianza**, más un ranking de oportunidades de revalorización del mercado.

## Arquitectura (separación de capas)

```
entregable4/
├── entrenar_modelo.py     # genera los artefactos (modelo + tabla de jugadores)
├── api/
│   ├── modelo.py          # LÓGICA DEL MODELO: carga artefactos, predice, rankea
│   └── main.py            # CAPA DE SERVICIO: API REST (FastAPI), entradas/salidas, errores
├── interfaz/
│   └── app.py             # INTERFAZ: Streamlit, consume la API por HTTP
├── modelos/               # artefactos entrenados (.joblib, metadata.json, tabla_jugadores.csv)
├── requirements.txt
└── README.md
```

La interfaz **no toca el modelo directamente**: todo pasa por la API (REST), como pide la consigna.

## Cómo correr — opción rápida (un solo comando)

Desde la carpeta `entregable4/`:

```bash
pip install -r requirements.txt
python iniciar.py        # levanta API + interfaz y abre el navegador
```

## Cómo correr — manual (dos terminales)

Desde la carpeta `entregable4/`:

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (solo si faltan los artefactos) entrenar y exportar el modelo
python entrenar_modelo.py

# 3. Levantar la API  (terminal 1)
uvicorn api.main:app --reload
#    Documentación interactiva: http://127.0.0.1:8000/docs

# 4. Levantar la interfaz  (terminal 2)
streamlit run interfaz/app.py
#    Se abre en http://localhost:8501
```

## Endpoints de la API

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/salud` | Estado del servicio |
| GET | `/jugadores?q=palmer` | Busca jugadores por nombre |
| GET | `/predecir/{player_id}` | Valor a 12 meses (intervalo P10–P90) |
| POST | `/predecir` | Igual, con simulación de features (edad, contrato, …) |
| GET | `/ranking?direccion=suben&min_valor=5000000` | Oportunidades de subida/baja |

**Ejemplo de salida de `/predecir/{id}`:**
```json
{
  "nombre": "Cole Palmer", "valor_actual": 15000000,
  "valor_estimado": 45200000,
  "intervalo": {"p10": 29000000, "p90": 70000000},
  "crecimiento_pct": 201.3, "direccion": "sube"
}
```

## Despliegue público (opcional — "entorno accesible")

Para que cualquiera navegue la app desde una URL pública, hay un `Dockerfile` + `start.sh` que
corren API + interfaz en un solo contenedor (la interfaz queda pública en el puerto 7860 y consume
la API interna). Recomendado: **Hugging Face Spaces** (gratis).

1. Crear un Space en https://huggingface.co/new-space → SDK: **Docker**.
2. Subir el contenido de `entregable4/` (incluida la carpeta `modelos/` con los artefactos).
3. HF construye la imagen y publica la URL → los profesores entran y la usan sin instalar nada.

(También sirve en Render, Railway o cualquier host con Docker.)

## Notas
- El modelo predice el **crecimiento** del valor y reconstruye el valor a 12 meses; el intervalo
  está **calibrado** (conformal) para contener el valor real ≈80% de las veces.
- Incluye el fix del Entregable 3 (orden temporal correcto en la validación).
- El valor de referencia es la estimación editorial de Transfermarkt, no un precio de venta real.
