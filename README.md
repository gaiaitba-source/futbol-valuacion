# ⚽ Scout AI — Predicción del Valor de Mercado de Futbolistas

Proyecto de **Ciencia de Datos Aplicada — ITBA**.
**Alumnos:** Octavio Argonz y Matías Sola.

Predecimos el **valor de mercado de un jugador a 12 meses** (con un rango de confianza) a partir
del dataset público [*Football Data from Transfermarkt*](https://www.kaggle.com/datasets/davidcariboo/player-scores).
El sistema incluye una **API REST** y una **interfaz web** que la consume (scouting de jugadores,
evolución del mercado, ranking de oportunidades y resúmenes con IA).

---

## 📂 Estructura

```
futbol-valuacion/
├── notebooks/                  # Entregables 2 y 3 (EDA + modelado)
│   └── Entregable3_Modelado_Futbol.ipynb
├── entregable4/                # Entregable 4 — despliegue (API + interfaz)
│   ├── api/                    #   FastAPI (servicio) + lógica de modelo + brief IA
│   ├── interfaz/               #   Streamlit (interfaz que consume la API)
│   ├── modelos/                #   modelo entrenado + tablas (listos para usar)
│   ├── iniciar.py              #   levanta todo con un comando
│   ├── Dockerfile / start.sh   #   para despliegue público
│   └── README.md               #   detalle del Entregable 4
├── docs/                       # consignas y documentación
└── data/raw/                   # CSVs del dataset (NO incluidos en el repo, ver abajo)
```

---

## 🚀 Cómo usar la aplicación (Entregable 4)

No hace falta el dataset ni reentrenar: el modelo ya viene entrenado en `entregable4/modelos/`.

```bash
cd entregable4
pip install -r requirements.txt
python iniciar.py
```

Eso levanta la API (puerto 8000) y la interfaz (puerto 8501) y abre el navegador en
**http://localhost:8501**. Listo para usar.

> **IA (opcional):** para que los resúmenes los escriba una IA, poné una API key gratuita de
> [Groq](https://console.groq.com) en `entregable4/.env`:
> ```
> GROQ_API_KEY=tu_clave
> ```
> Sin la key, igual funciona con un resumen automático.

---

## 📊 Qué hace el modelo (resumen)

- Predice el **crecimiento** del valor (no el nivel), para que el modelo aprenda de edad,
  rendimiento, liga, club, contrato y transferencias — no solo del valor actual.
- Horizonte **anclado a 12 meses**; corrige el **sesgo de supervivencia** (aprende también las caídas).
- Entrega un **intervalo calibrado** (~80% de cobertura) + un **ranking de oportunidades**.
- Mejor modelo: **XGBoost** — R² nivel 0.88 (baseline 0.82), MAE €0.75M.

---

## 📥 Sobre los datos

Los CSVs (`data/raw/`) pesan ~700 MB y **no se versionan**. Solo son necesarios para
**reentrenar** el modelo (`entregable4/entrenar_modelo.py`) o correr los notebooks. Para usar la
app no hacen falta. Se descargan de [Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)
y se colocan en `data/raw/`.
