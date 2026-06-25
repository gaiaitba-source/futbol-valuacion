"""
Capa de SERVICIO — API REST (FastAPI).

Expone el modelo de valuación a través de endpoints HTTP. La lógica vive en modelo.py;
acá solo definimos entradas/salidas, validación y manejo de errores.

Levantar:  uvicorn api.main:app --reload   (desde la carpeta entregable4/)
Docs:      http://127.0.0.1:8000/docs
"""
import os
from pathlib import Path as _Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def _cargar_env():
    """Carga variables desde entregable4/.env (ej. GROQ_API_KEY) sin dependencias externas."""
    f = _Path(__file__).resolve().parent.parent / '.env'
    if f.exists():
        for line in f.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_cargar_env()

from .modelo import ModeloValuacion
from .brief import generar_brief
from .chat import responder as chat_responder

app = FastAPI(
    title="API — Valuación de futbolistas a 12 meses",
    description="Estima el valor de mercado futuro de un jugador (con intervalo de confianza) "
                "y rankea oportunidades de revalorización. Entregable 4 — ITBA.",
    version="2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    modelo = ModeloValuacion()
except Exception as e:  # pragma: no cover
    modelo = None
    _error_carga = str(e)


class Overrides(BaseModel):
    edad: float | None = None
    contrato_restante: float | None = None
    n_ap: float | None = None
    goles_por_partido: float | None = None
    asist_por_partido: float | None = None
    minutos_promedio: float | None = None


class PedidoPrediccion(BaseModel):
    player_id: int = Field(..., description="ID del jugador (de /jugadores)")
    overrides: Overrides | None = Field(None, description="Simulación: cambiar features puntuales")


def _check():
    if modelo is None:
        raise HTTPException(503, f"Modelo no disponible: {_error_carga}")


@app.get("/salud")
def salud():
    """Estado del servicio."""
    _check()
    return {"estado": "ok", "jugadores_en_catalogo": len(modelo.tabla),
            "features": len(modelo.features)}


@app.get("/opciones")
def opciones():
    """Opciones para los filtros de la interfaz (posiciones, ligas, países, tiers)."""
    _check()
    return modelo.opciones()


@app.get("/jugadores")
def jugadores(q: str = Query(..., min_length=2, description="Texto a buscar en el nombre"),
              posicion: str | None = None, liga: str | None = None, pais: str | None = None):
    """Busca jugadores por nombre, con filtros opcionales."""
    _check()
    res = modelo.buscar(q, posicion=posicion, liga=liga, pais=pais)
    if not res:
        return {"resultados": [], "mensaje": f"Sin coincidencias para '{q}'"}
    return {"resultados": res}


@app.get("/predecir/{player_id}")
def predecir(player_id: int):
    """Predice el valor a 12 meses de un jugador (intervalo P10–P90)."""
    _check()
    try:
        return modelo.predecir(player_id)
    except KeyError:
        raise HTTPException(404, f"No existe el jugador con id {player_id}")
    except Exception as e:  # noqa
        raise HTTPException(500, f"Error al predecir: {e}")


@app.post("/predecir")
def predecir_con_overrides(pedido: PedidoPrediccion):
    """Predicción con simulación: permite cambiar features (edad, contrato, etc.)."""
    _check()
    ov = pedido.overrides.model_dump(exclude_none=True) if pedido.overrides else None
    try:
        return modelo.predecir(pedido.player_id, overrides=ov)
    except KeyError:
        raise HTTPException(404, f"No existe el jugador con id {pedido.player_id}")
    except Exception as e:  # noqa
        raise HTTPException(500, f"Error al predecir: {e}")


@app.get("/historico/{player_id}")
def historico(player_id: int):
    """Trayectoria histórica REAL del jugador (valor por año y cambio observado a 12 meses)."""
    _check()
    try:
        return modelo.historico_jugador(player_id)
    except KeyError:
        raise HTTPException(404, f"No hay histórico para el jugador {player_id}")


class MensajeChat(BaseModel):
    role: str
    content: str


class PedidoChat(BaseModel):
    mensajes: list[MensajeChat] = Field(..., description="Historial de la conversación (user/assistant)")


@app.post("/chat")
def chat(p: PedidoChat):
    """Asistente de scouting: recibe la conversación y devuelve respuesta + candidatos."""
    _check()
    msgs = [{"role": m.role, "content": m.content} for m in p.mensajes]
    return chat_responder(msgs, modelo)


@app.get("/brief/{player_id}")
def brief(player_id: int):
    """Resumen de scouting en lenguaje natural (IA gratuita si hay API key, si no plantilla)."""
    _check()
    try:
        pred = modelo.predecir(player_id)
        serie = modelo.historico_jugador(player_id)["serie"]
        datos = {**pred, "pico": max(s["valor"] for s in serie),
                 "anio_desde": serie[0]["anio"], "anio_hasta": serie[-1]["anio"]}
        return generar_brief(datos)
    except KeyError:
        raise HTTPException(404, f"No existe el jugador con id {player_id}")
    except Exception as e:  # noqa
        raise HTTPException(500, f"Error al generar el brief: {e}")


@app.get("/clubs")
def clubs(liga: str | None = None, pais: str | None = None):
    """Lista de clubes, opcionalmente filtrada por liga y/o nacionalidad (filtros dependientes)."""
    _check()
    return {"clubs": modelo.clubs_de(liga, pais)}


@app.get("/evolucion")
def evolucion(posicion: str | None = None, liga: str | None = None,
              pais: str | None = None, club: str | None = None, player_id: int | None = None):
    """Evolución agregada del mercado (valor mediano por año) + resumen, con filtros opcionales."""
    _check()
    return modelo.evolucion(posicion=posicion, liga=liga, pais=pais, club=club, player_id=player_id)


@app.get("/ranking")
def ranking(direccion: str = Query("suben", pattern="^(suben|bajan)$"),
            tier: str = Query("consolidados", pattern="^(estrellas|consolidados|promesas|todos)$"),
            limite: int = Query(12, ge=1, le=50),
            posicion: str | None = None, liga: str | None = None, pais: str | None = None):
    """Ranking de oportunidades: jugadores que el modelo proyecta que suben o bajan."""
    _check()
    return {"direccion": direccion, "tier": tier,
            "resultados": modelo.ranking(direccion, tier, limite, posicion, liga, pais)}
