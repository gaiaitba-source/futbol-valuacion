"""
Generación de un BRIEF de scouting en lenguaje natural.

Usa una IA gratuita si hay una API key configurada por variable de entorno:
  - GROQ_API_KEY   (Groq, modelos Llama — free tier sin tarjeta, recomendado)
  - GEMINI_API_KEY (Google Gemini)
Si no hay key (o la llamada falla), cae a un resumen automático armado con los datos
(plantilla), de modo que la funcionalidad SIEMPRE responde.
"""
import os
import requests


def _eur(x: float) -> str:
    return f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K"


def _recomendacion(d: dict):
    """Veredicto tipo app de inversiones (5 niveles). Devuelve (etiqueta, nivel)."""
    c = d["crecimiento_pct"]
    if c >= 25:
        return ("COMPRA FUERTE", "compra_fuerte")
    if c >= 8:
        return ("COMPRA", "compra")
    if c <= -25:
        return ("VENTA FUERTE", "venta_fuerte")
    if c <= -8:
        return ("VENTA", "venta")
    return ("NEUTRAL · RETENER", "neutral")


def _prompt(d: dict, reco: str) -> str:
    return (
        "Sos un analista de scouting de fútbol profesional. Escribí un análisis breve (4-5 frases), "
        "en español neutro y claro, que: (1) presente al jugador, (2) interprete la proyección del "
        "modelo y su trayectoria, y (3) cierre con una RECOMENDACIÓN accionable para un club "
        "(comprar, retener o vender/evitar) con una justificación corta. No inventes datos: usá solo "
        "los que te paso. El modelo sugiere a priori: "
        f"'{reco}'. Sé concreto y profesional.\n\n"
        f"Jugador: {d['nombre']}\n"
        f"Posición: {d['posicion']} | Edad: {d['edad']} | Nacionalidad: {d['pais']}\n"
        f"Liga/club: {d['liga']} / {d.get('club','—')}\n"
        f"Valor actual ({d['fecha_base']}): {_eur(d['valor_actual'])}\n"
        f"Valor estimado por el modelo ({d['fecha_objetivo']}): {_eur(d['valor_estimado'])} "
        f"({d['crecimiento_pct']:+.0f}%), rango {_eur(d['intervalo']['p10'])}–{_eur(d['intervalo']['p90'])}\n"
        f"Tendencia proyectada: {d['direccion']}\n"
        f"Pico histórico de valor: {_eur(d['pico'])} | historial {d['anio_desde']}–{d['anio_hasta']}\n"
    )


def _groq(prompt: str, key: str) -> str:
    modelo = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": modelo,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.6, "max_tokens": 320},
        timeout=25)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _gemini(prompt: str, key: str) -> str:
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}",
        json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _fallback(d: dict, reco: str) -> str:
    dir_txt = {"sube": "una revalorización", "baja": "una caída de valor",
               "estable": "estabilidad en su valor"}[d["direccion"]]
    return (
        f"{d['nombre']} es un {d['posicion'].lower()} de {d['edad']} años "
        f"({d['pais']}) que juega en {d['liga']}. Su valor de mercado actual es "
        f"{_eur(d['valor_actual'])}. El modelo proyecta {dir_txt}: lo estima en "
        f"{_eur(d['valor_estimado'])} para {d['fecha_objetivo']} ({d['crecimiento_pct']:+.0f}%), "
        f"con un rango probable de {_eur(d['intervalo']['p10'])} a {_eur(d['intervalo']['p90'])}. "
        f"A lo largo de su carrera alcanzó un pico de {_eur(d['pico'])}. "
        f"Recomendación: {reco}.")


def generar_brief(d: dict) -> dict:
    """Devuelve {'texto', 'fuente', 'recomendacion', 'nivel'}."""
    etiqueta, nivel = _recomendacion(d)
    prompt = _prompt(d, etiqueta)
    key_groq, key_gem = os.environ.get("GROQ_API_KEY"), os.environ.get("GEMINI_API_KEY")
    base = {"recomendacion": etiqueta, "nivel": nivel}
    try:
        if key_groq:
            return {**base, "texto": _groq(prompt, key_groq),
                    "fuente": f"IA · Groq ({os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')})"}
        if key_gem:
            return {**base, "texto": _gemini(prompt, key_gem), "fuente": "IA · Gemini 1.5 Flash"}
    except Exception:  # noqa
        return {**base, "texto": _fallback(d, etiqueta), "fuente": "resumen automático (IA no disponible)"}
    return {**base, "texto": _fallback(d, etiqueta), "fuente": "resumen automático (sin API key de IA)"}
