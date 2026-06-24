"""
Lanzador todo-en-uno (entorno local).

Levanta la API (FastAPI/uvicorn) y la interfaz (Streamlit) con un solo comando, y abre
el navegador. Pensado para que cualquiera —por ejemplo, los profesores— pueda probar la
solución sin conocer los comandos.

Uso:   python iniciar.py     (desde la carpeta entregable4/)
Cortar: Ctrl+C
"""
import subprocess, sys, time, webbrowser
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PY = sys.executable

print("⏳ Levantando la API en http://127.0.0.1:8000 ...")
api = subprocess.Popen([PY, "-m", "uvicorn", "api.main:app", "--port", "8000"], cwd=AQUI)

# Esperar a que la API responda
import urllib.request
for _ in range(40):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/salud", timeout=2)
        print("✅ API lista.")
        break
    except Exception:
        time.sleep(1)
else:
    print("⚠️ La API tardó en responder; sigo igual.")

print("⏳ Abriendo la interfaz en http://localhost:8501 ...")
try:
    webbrowser.open("http://localhost:8501")
except Exception:
    pass

try:
    # Streamlit en primer plano (bloquea hasta Ctrl+C)
    subprocess.run([PY, "-m", "streamlit", "run", "interfaz/app.py",
                    "--server.port", "8501"], cwd=AQUI)
finally:
    print("\n🛑 Cerrando la API...")
    api.terminate()
