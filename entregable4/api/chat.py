"""
Asistente de scouting conversacional (function calling con Groq/Llama).

El usuario pide en lenguaje natural. El modelo de lenguaje INDAGA (posición, presupuesto,
perfil, edad, físico) antes de recomendar, y cuando tiene lo necesario llama a la herramienta
`buscar_candidatos`, que consulta la tabla de jugadores y devuelve opciones según los criterios.
La IA no accede a internet: solo a los datos del modelo (la tabla servida).
"""
import os
import json
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SISTEMA = (
    "Sos el asistente de scouting de Scout AI. Ayudás a un club a encontrar jugadores. "
    "Solo trabajás con los datos del modelo (no tenés acceso a internet). Reglas:\n"
    "- NO asumas la posición. Si el usuario no la indicó, preguntala. Nunca llames a la herramienta sin posición.\n"
    "- Indagá para entender bien la necesidad: además de posición y presupuesto, preguntá (en 1-2 mensajes, "
    "sin abrumar) si tiene preferencias de PERFIL (¿prioriza proyección a futuro o experiencia/jerarquía?), "
    "rango de EDAD, o requisitos FÍSICOS (ej. altura mínima para un central o un 9).\n"
    "- Para recomendar SIEMPRE usá la herramienta buscar_candidatos. No inventes jugadores ni valores.\n"
    "- La herramienta ya busca jugadores de valor CERCANO al presupuesto (hasta 20% por debajo). "
    "Para perfiles de experiencia/jerarquía usá orden='experiencia'; para apuestas a futuro, orden='proyeccion'.\n"
    "- NO fuerces jugadores al alza: si el usuario busca experiencia, está bien recomendar aunque el valor no suba.\n"
    "- Recomendá las 3 mejores opciones con: valor actual, valor estimado a 12 meses y el cambio %. "
    "Explicá en 1 línea por qué encaja con lo que pidió.\n"
    "- Aclará que los valores son estimaciones editoriales de Transfermarkt (no precios de venta), a 12 meses.\n"
    "- Respondé en español, conciso y concreto."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "buscar_candidatos",
        "description": "Busca jugadores en la base según criterios. Devuelve candidatos con su "
                       "proyección de valor a 12 meses. Requiere que el usuario haya indicado la posición.",
        "parameters": {
            "type": "object",
            "properties": {
                "posicion": {"type": "string", "enum": ["Arquero", "Defensor", "Mediocampista", "Delantero"]},
                "presupuesto_max_eur": {"type": "number", "description": "Presupuesto, en euros (ej. 85000000)"},
                "valor_min_eur": {"type": "number", "description": "Valor mínimo en euros. Usar 0 si busca gangas/promesas."},
                "edad_min": {"type": "integer"},
                "edad_max": {"type": "integer"},
                "altura_min_cm": {"type": "number", "description": "Altura mínima en cm (ej. 185)"},
                "liga": {"type": "string", "description": "Texto a buscar en la liga (ej. 'Premier', 'LaLiga')"},
                "orden": {"type": "string", "enum": ["proyeccion", "experiencia"],
                          "description": "proyeccion = mayor crecimiento esperado; experiencia = jugadores de mayor valor/jerarquía"},
            },
            "required": ["posicion"],
        },
    },
}]


def _ejecutar(modelo, args: dict) -> list:
    df = modelo.tabla[modelo.tabla["anio"] >= 2024].copy()
    if args.get("posicion"):
        df = df[df["posicion"] == args["posicion"]]
    if args.get("edad_min"):
        df = df[df["edad"] >= int(args["edad_min"])]
    if args.get("edad_max"):
        df = df[df["edad"] <= int(args["edad_max"])]
    if args.get("altura_min_cm") and "height_in_cm" in df:
        df = df[df["height_in_cm"] >= float(args["altura_min_cm"])]
    if args.get("liga"):
        df = df[df["liga_display"].str.contains(str(args["liga"]), case=False, na=False)]

    pmax = args.get("presupuesto_max_eur")
    if pmax:
        df = df[df["valor_actual"] <= float(pmax)]
    # Banda de valor: cerca del presupuesto (hasta -20%), con fallback si quedan pocos.
    vmin = args.get("valor_min_eur")
    if vmin is not None:
        df = df[df["valor_actual"] >= max(1_000_000, float(vmin))]
    elif pmax:
        for frac in (0.8, 0.6, 0.4, 0.2, 0.0):
            cand = df[df["valor_actual"] >= frac * float(pmax)]
            if len(cand) >= 4 or frac == 0.0:
                df = cand
                break
    else:
        df = df[df["valor_actual"] >= 1_000_000]

    if args.get("orden") == "experiencia":
        df = df.sort_values("valor_actual", ascending=False)
    else:
        df = df.sort_values("crecimiento_pct", ascending=False)
    return [{"player_id": int(r.player_id), "nombre": r.name, "posicion": r.posicion, "edad": int(r.edad),
             "liga": r.liga_display, "valor_actual_M": round(r.valor_actual / 1e6, 1),
             "valor_estimado_M": round(r.p50 / 1e6, 1), "crecimiento_pct": round(float(r.crecimiento_pct), 1)}
            for r in df.head(8).itertuples()]


def _ordenar_por_texto(texto: str, cands: list) -> list:
    """Reordena los candidatos según el orden en que aparecen en la respuesta de la IA
    (los recomendados quedan primero, en el mismo orden que el texto)."""
    t = (texto or "").lower()
    return sorted(cands, key=lambda c: (t.find(c["nombre"].lower()) if c["nombre"].lower() in t else 10**9))


def responder(mensajes: list, modelo) -> dict:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return {"respuesta": "El asistente necesita una API key de IA configurada (GROQ_API_KEY).", "candidatos": []}
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    msgs = [{"role": "system", "content": SISTEMA}] + mensajes
    candidatos = []
    try:
        for _ in range(3):
            payload = {"model": model, "messages": msgs, "tools": TOOLS, "temperature": 0.5, "max_tokens": 750}
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=35)
            r.raise_for_status()
            m = r.json()["choices"][0]["message"]
            if m.get("tool_calls"):
                msgs.append(m)
                for tc in m["tool_calls"]:
                    try:
                        a = json.loads(tc["function"]["arguments"] or "{}")
                    except Exception:
                        a = {}
                    candidatos = _ejecutar(modelo, a)
                    msgs.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(candidatos, ensure_ascii=False)})
                continue
            txt = (m.get("content") or "").strip()
            return {"respuesta": txt, "candidatos": _ordenar_por_texto(txt, candidatos)}
        return {"respuesta": "No pude completar la recomendación, probá reformular el pedido.", "candidatos": candidatos}
    except Exception as e:  # noqa
        return {"respuesta": f"Hubo un problema consultando la IA: {e}", "candidatos": candidatos}
