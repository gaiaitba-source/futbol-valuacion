"""
INTERFAZ DE USO — Streamlit (versión visual retro/grunge + navegación + brief IA).

Navegación por páginas (controlada por código) para poder saltar del ranking a la ficha de un
jugador. Cada ficha muestra: trayectoria histórica real, predicción a 12 meses con rango y fecha,
simulador de escenarios y un resumen de scouting generado por IA (vía la API). Todo consume la API.

Levantar:  streamlit run interfaz/app.py     (con la API en el puerto 8000)
Fondo propio: poné tu imagen en  interfaz/assets/fondo.jpg
"""
import os, base64, math
from pathlib import Path
import requests
import pandas as pd
import altair as alt
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
AQUI = Path(__file__).resolve().parent
st.set_page_config(page_title="Scout AI — Valuación de futbolistas", page_icon="⚽", layout="wide")
AZUL, CELESTE, ORO, VERDE, ROJO = "#6cb4ee", "#8fd0ff", "#e6c15a", "#39d98a", "#ff6b6b"


def eur(x: float) -> str:
    return f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K"


# --- estado de navegación ---
st.session_state.setdefault("page", "consultar")
st.session_state.setdefault("pid", None)


def ir_a(page, pid=None):
    st.session_state.page = page
    if pid is not None:
        st.session_state.pid = pid


def _fondo_css() -> str:
    img = AQUI / "assets" / "fondo.jpg"
    if img.exists():
        b64 = base64.b64encode(img.read_bytes()).decode()
        bg = (f"linear-gradient(rgba(6,16,28,.80),rgba(6,14,26,.93)),"
              f"url('data:image/jpeg;base64,{b64}')")
        extra = "background-size:cover;background-position:center top;background-attachment:fixed;"
    else:
        bg = "radial-gradient(1200px 600px at 50% -10%, #18476f 0%, #0c2740 45%, #06121f 100%)"
        extra = ""
    return f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;800&display=swap');
      .stApp {{ background:{bg}; {extra} font-family:'Inter',sans-serif; }}
      .block-container {{ padding-top:1.3rem; max-width:1250px; }}
      h1,h2,h3,h4 {{ font-family:'Oswald',sans-serif; letter-spacing:.4px; color:#eaf3fc !important; }}
      p,span,label,div,li {{ color:#dbe7f3; }}
      .hero {{ position:relative; padding:28px 34px; border-radius:20px; margin-bottom:14px;
        background:linear-gradient(120deg, rgba(108,180,238,.18), rgba(230,193,90,.10));
        border:1px solid rgba(143,208,255,.22); box-shadow:0 14px 40px rgba(0,0,0,.5); }}
      .hero h1 {{ font-family:'Anton',sans-serif !important; margin:0; font-size:46px; line-height:1;
        color:#fff !important; text-transform:uppercase; letter-spacing:1px; text-shadow:0 3px 18px rgba(0,0,0,.6); }}
      .hero h1 .oro {{ color:{ORO}; }}
      .hero p {{ margin:.5rem 0 0; color:#c9dcef; font-size:15px; }}
      .hero::after {{ content:''; position:absolute; left:34px; bottom:14px; width:80px; height:5px;
        background:{ORO}; border-radius:3px; box-shadow:0 0 18px rgba(230,193,90,.7); }}
      .card {{ background:rgba(11,28,46,.74); border:1px solid rgba(143,208,255,.16); border-radius:18px;
        padding:20px 24px; margin-bottom:14px; box-shadow:0 12px 34px rgba(0,0,0,.45); backdrop-filter:blur(4px); }}
      .briefcard {{ background:linear-gradient(135deg, rgba(230,193,90,.10), rgba(11,28,46,.8));
        border:1px solid rgba(230,193,90,.3); border-radius:18px; padding:18px 22px; margin-bottom:14px;
        box-shadow:0 12px 34px rgba(0,0,0,.45); }}
      .briefcard p {{ font-size:16.5px; line-height:1.62; }}
      .pill {{ display:inline-block; padding:4px 14px; border-radius:999px; font-size:12px; font-weight:800;
        font-family:'Oswald'; text-transform:uppercase; letter-spacing:.5px; }}
      .rk {{ background:rgba(11,28,46,.66); border:1px solid rgba(143,208,255,.12); border-left:5px solid var(--c,{VERDE});
        border-radius:14px; padding:11px 16px; box-shadow:0 6px 18px rgba(0,0,0,.35); }}
      div[data-testid="stMetric"] {{ background:rgba(11,28,46,.7); border:1px solid rgba(143,208,255,.16);
        border-radius:14px; padding:12px 16px; box-shadow:0 8px 22px rgba(0,0,0,.4); }}
      div[data-testid="stMetricValue"] {{ font-family:'Oswald'; color:#fff; }}
      .stTextInput input, div[data-baseweb="select"] > div {{ background:rgba(8,20,34,.85) !important;
        border:1px solid rgba(143,208,255,.22) !important; border-radius:10px !important; color:#eaf3fc !important; }}
      .stButton button {{ font-family:'Oswald'; font-weight:600; letter-spacing:.4px; border-radius:12px;
        background:linear-gradient(120deg,{AZUL},#3f8fd0); border:0; color:#06121f;
        box-shadow:0 8px 22px rgba(108,180,238,.30); }}
      .stButton button:disabled {{ background:rgba(230,193,90,.92); color:#06121f; opacity:1; }}
      div[data-testid="stVegaLiteChart"], .stVegaLiteChart {{ background:rgba(8,20,34,.55);
        border:1px solid rgba(143,208,255,.16); border-radius:16px; padding:12px 10px; box-shadow:0 12px 30px rgba(0,0,0,.45); }}
    </style>"""


st.markdown(_fondo_css(), unsafe_allow_html=True)


def _cfg(ch):
    return (ch.properties(height=300).configure_view(strokeWidth=0, fill='transparent')
            .configure_axis(labelColor='#a9c4dd', titleColor='#a9c4dd', labelFont='Inter', titleFont='Oswald',
                            gridColor='rgba(143,208,255,.10)', domainColor='#33506e', tickColor='#33506e')
            .configure_legend(labelColor='#dbe7f3', titleColor='#dbe7f3', labelFont='Inter'))


def grafico_jugador(serie):
    df = pd.DataFrame(serie); df["Valor (M€)"] = df["valor"] / 1e6
    ch = alt.Chart(df).mark_line(color=CELESTE, strokeWidth=3,
            point=alt.OverlayMarkDef(color=ORO, size=70, filled=True, stroke="#06121f")).encode(
        x=alt.X("anio:O", title=None), y=alt.Y("Valor (M€):Q", title="Valor (M€)"),
        tooltip=[alt.Tooltip("anio:O", title="Año"), alt.Tooltip("Valor (M€):Q", format=".1f")])
    return _cfg(ch)


def grafico_mercado(serie):
    df = pd.DataFrame(serie); df["Mediano"] = df["valor_mediano"]/1e6; df["Promedio"] = df["valor_promedio"]/1e6
    dl = df.melt(id_vars="anio", value_vars=["Mediano", "Promedio"], var_name="Serie", value_name="M€")
    ch = alt.Chart(dl).mark_line(strokeWidth=3, point=alt.OverlayMarkDef(filled=True, size=55)).encode(
        x=alt.X("anio:O", title=None), y=alt.Y("M€:Q", title="Valor (M€)"),
        color=alt.Color("Serie:N", scale=alt.Scale(domain=["Mediano", "Promedio"], range=[CELESTE, ORO]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("anio:O", title="Año"), "Serie", alt.Tooltip("M€:Q", format=".1f")])
    return _cfg(ch)


def nivel_de(c):
    if c >= 25: return ("COMPRA FUERTE", "compra_fuerte")
    if c >= 8:  return ("COMPRA", "compra")
    if c <= -25: return ("VENTA FUERTE", "venta_fuerte")
    if c <= -8:  return ("VENTA", "venta")
    return ("NEUTRAL · RETENER", "neutral")


def gauge_svg(crec) -> str:
    """Velocímetro de recomendación tipo app de inversiones (Venta ← → Compra)."""
    etiqueta, nivel = nivel_de(crec)
    cx, cy, R, W, r = 150, 150, 116, 24, 96
    segs = [(180, 144, "#e74c3c"), (144, 108, "#e67e22"), (108, 72, "#9aa7b3"),
            (72, 36, "#7fce8e"), (36, 0, "#2ecc71")]
    pt = lambda t, rad: (cx + rad * math.cos(math.radians(t)), cy - rad * math.sin(math.radians(t)))
    arcs = ""
    for a1, a2, c in segs:
        x1, y1 = pt(a1, R); x2, y2 = pt(a2, R)
        arcs += (f'<path d="M {x1:.1f} {y1:.1f} A {R} {R} 0 0 1 {x2:.1f} {y2:.1f}" '
                 f'stroke="{c}" stroke-width="{W}" fill="none"/>')
    cc = max(-40, min(40, crec)); th = 90 - (cc / 40) * 90; nx, ny = pt(th, r)
    color = {"compra_fuerte": "#2ecc71", "compra": "#7fce8e", "neutral": "#e6c15a",
             "venta": "#e67e22", "venta_fuerte": "#e74c3c"}[nivel]
    return (f'<div style="text-align:center;margin:4px 0 2px"><svg viewBox="0 0 300 192" '
            f'width="100%" style="max-width:340px">{arcs}'
            f'<text x="30" y="170" fill="#e7867a" font-family="Oswald" font-size="11">VENTA</text>'
            f'<text x="270" y="170" fill="#8fe3ad" font-family="Oswald" font-size="11" text-anchor="end">COMPRA</text>'
            f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#eaf3fc" stroke-width="5" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="10" fill="#eaf3fc"/>'
            f'<text x="{cx}" y="188" text-anchor="middle" fill="{color}" font-family="Oswald" '
            f'font-weight="700" font-size="23">{etiqueta}</text></svg></div>')


def barra_intervalo(valor, p10, p50, p90) -> str:
    lo, hi = min(p10, valor)*0.95, max(p90, valor)*1.05
    span = max(hi - lo, 1); pos = lambda v: max(0, min(100, (v-lo)/span*100))
    a, b, p, v = pos(p10), pos(p90), pos(p50), pos(valor)
    return f"""<div style="margin:16px 0 4px;"><div style="position:relative;height:50px;">
      <div style="position:absolute;top:21px;left:0;width:100%;height:11px;background:#13314f;border-radius:6px;"></div>
      <div style="position:absolute;top:21px;left:{a}%;width:{max(b-a,1)}%;height:11px;
           background:linear-gradient(90deg,{AZUL},{ORO});border-radius:6px;box-shadow:0 0 14px rgba(230,193,90,.4);"></div>
      <div style="position:absolute;top:11px;left:{p}%;transform:translateX(-50%);font-size:22px;">⚪</div>
      <div style="position:absolute;top:0;left:{v}%;transform:translateX(-50%);color:{ORO};font-weight:800;
           font-size:12px;white-space:nowrap;font-family:Oswald;">▼ HOY</div></div>
      <div style="display:flex;justify-content:space-between;font-size:13px;color:#a9c4dd;font-family:Oswald;">
        <span>PESIMISTA<br><b style="color:#eaf3fc">{eur(p10)}</b></span>
        <span style="text-align:center">ESTIMADO<br><b style="color:{VERDE}">{eur(p50)}</b></span>
        <span style="text-align:right">OPTIMISTA<br><b style="color:#eaf3fc">{eur(p90)}</b></span></div></div>"""


try:
    salud = requests.get(f"{API}/salud", timeout=5).json()
except Exception:
    st.error(f"No me conecto a la API en {API}. ¿Está corriendo `uvicorn api.main:app`?")
    st.stop()


@st.cache_data(ttl=300)
def get_opciones():
    return requests.get(f"{API}/opciones", timeout=10).json()


@st.cache_data(ttl=300)
def get_clubs(liga, pais):
    params = {k: v for k, v in [("liga", liga), ("pais", pais)] if v}
    return requests.get(f"{API}/clubs", params=params, timeout=10).json()["clubs"]


op = get_opciones()
st.markdown(f"""<div class="hero"><h1>⚽ Scout <span class="oro">AI</span></h1>
  <p>Cuánto valdrá un jugador <b>dentro de 12 meses</b> — con rango de confianza, su historia real
  y la evolución del mercado. &nbsp;·&nbsp; {salud['jugadores_en_catalogo']:,} jugadores.</p></div>""",
  unsafe_allow_html=True)

# --- Navegación (botones; el activo queda resaltado/deshabilitado) ---
n1, n2, n3, n4 = st.columns(4)
n1.button("🔎  CONSULTAR", use_container_width=True, disabled=st.session_state.page == "consultar",
          on_click=ir_a, args=("consultar",))
n2.button("📈  MERCADO", use_container_width=True, disabled=st.session_state.page == "evolucion",
          on_click=ir_a, args=("evolucion",))
n3.button("🏆  OPORTUNIDADES", use_container_width=True, disabled=st.session_state.page == "ranking",
          on_click=ir_a, args=("ranking",))
n4.button("💬  ASISTENTE IA", use_container_width=True, disabled=st.session_state.page == "asistente",
          on_click=ir_a, args=("asistente",))
st.write("")


# ====================================================== PÁGINA: CONSULTAR =====
def pagina_consultar():
    c1, c2, c3, c4 = st.columns([2.2, 1, 1.4, 1.4])
    q = c1.text_input("Jugador", placeholder="Palmer, Mbappé, Messi...")
    f_pos = c2.selectbox("Posición", ["Todas"] + op["posiciones"])
    f_liga = c3.selectbox("Liga", ["Todas"] + op["ligas"])
    f_pais = c4.selectbox("Nacionalidad", ["Todas"] + op["paises"])

    pid = None
    if q and len(q) >= 2:
        params = {"q": q}
        if f_pos != "Todas": params["posicion"] = f_pos
        if f_liga != "Todas": params["liga"] = f_liga
        if f_pais != "Todas": params["pais"] = f_pais
        res = requests.get(f"{API}/jugadores", params=params, timeout=10).json().get("resultados", [])
        if not res:
            st.warning("Sin coincidencias con esos filtros."); return
        etq = {f"{o['nombre']} — {o['posicion']}, {o['edad']}a · {o['liga']} ({eur(o['valor_actual'])})": o for o in res}
        jug = etq[st.selectbox("Resultados", list(etq.keys()))]
        pid = jug["player_id"]; st.session_state.pid = pid
    elif st.session_state.pid:
        pid = st.session_state.pid
    else:
        st.info("Escribí al menos 2 letras para buscar, o entrá desde el ranking de Oportunidades.")
        return

    base = requests.get(f"{API}/predecir/{pid}", timeout=10).json()

    # Trayectoria real
    serie = requests.get(f"{API}/historico/{pid}", timeout=10).json().get("serie", [])
    if len(serie) >= 2:
        st.markdown("#### 📉 Trayectoria real del valor de mercado")
        st.altair_chart(grafico_jugador(serie), use_container_width=True)
        cambios = [s for s in serie if s["cambio_real_pct"] is not None]
        if cambios:
            prom = sum(s["cambio_real_pct"] for s in cambios) / len(cambios)
            st.caption(f"Pico histórico: {eur(max(s['valor'] for s in serie))} · cambio anual real promedio: {prom:+.0f}%")

    with st.expander("🎛️ Simular un escenario hipotético (qué pasaría si…)"):
        s1, s2 = st.columns(2)
        ue = s1.checkbox("Cambiar edad")
        edad = s1.slider("Edad", 15, 40, int(base["edad"]), disabled=not ue)
        uc = s2.checkbox("Cambiar años de contrato")
        contrato = s2.slider("Años de contrato restantes", 0, 6, 2, disabled=not uc)
    ov = {}
    if ue: ov["edad"] = edad
    if uc: ov["contrato_restante"] = contrato

    pred = base
    if ov and st.button("📊  SIMULAR ESCENARIO", type="primary"):
        pred = requests.post(f"{API}/predecir", json={"player_id": pid, "overrides": ov}, timeout=10).json()

    color = {"sube": VERDE, "baja": ROJO, "estable": ORO}[pred["direccion"]]
    icono = {"sube": "📈", "baja": "📉", "estable": "➡️"}[pred["direccion"]]
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-top:0'>{pred['nombre']} &nbsp;<span class='pill' style='background:{color};"
                f"color:#06121f'>{icono} {pred['direccion'].upper()}</span></h3>"
                f"<p style='color:#9fb3c8'>{pred['posicion']} · {pred['edad']} años · {pred['liga']} · {pred['pais']}</p>",
                unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric(f"Valor base ({pred['fecha_base']})", eur(pred["valor_actual"]))
    m2.metric(f"Estimado ({pred['fecha_objetivo']})", eur(pred["valor_estimado"]), delta=f"{pred['crecimiento_pct']:+.0f}%")
    st.markdown(barra_intervalo(pred["valor_actual"], pred["intervalo"]["p10"], pred["valor_estimado"],
                                pred["intervalo"]["p90"]), unsafe_allow_html=True)
    st.caption(f"Rango con **{pred['confianza']*100:.0f}% de probabilidad** de contener el valor real "
               f"(predicción a {pred['fecha_objetivo']}). ⚪ = estimado · ▼ = valor de hoy.")
    st.markdown("<p style='text-align:center;color:#9fb3c8;margin:.4rem 0 0;font-family:Oswald'>"
                "RECOMENDACIÓN DEL MODELO</p>" + gauge_svg(pred["crecimiento_pct"]), unsafe_allow_html=True)
    if pred.get("es_simulacion"):
        st.info(f"⚠️ Escenario hipotético — simulación: {pred['overrides_aplicados']}")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Brief IA ---
    bkey = f"brief_{pid}"
    if st.button("🧠  Generar resumen de scouting (IA)"):
        with st.spinner("Generando resumen..."):
            st.session_state[bkey] = requests.get(f"{API}/brief/{pid}", timeout=30).json()
    if bkey in st.session_state:
        b = st.session_state[bkey]
        col_reco = {"compra_fuerte": VERDE, "compra": "#7fce8e", "neutral": ORO,
                    "venta": "#e67e22", "venta_fuerte": ROJO}.get(b.get("nivel"), ORO)
        badge = (f"<span class='pill' style='background:{col_reco};color:#06121f'>"
                 f"{b.get('recomendacion','—')}</span>") if b.get("recomendacion") else ""
        st.markdown(f"<div class='briefcard'><b style='font-family:Oswald;color:{ORO}'>🧠 ANÁLISIS DE SCOUTING</b> "
                    f"{badge}<p style='margin:.6rem 0 0'>{b['texto']}</p>"
                    f"<p style='color:#9fb3c8;font-size:12px;margin-top:.6rem'>Fuente: {b['fuente']}</p></div>",
                    unsafe_allow_html=True)


# ====================================================== PÁGINA: EVOLUCIÓN =====
def pagina_evolucion():
    st.markdown("Cómo evolucionó el **valor del mercado** a lo largo de los años. Sin filtros muestra el "
                "mercado completo; filtrá para ver un segmento.")
    c1, c2, c3, c4 = st.columns(4)
    e_pos = c1.selectbox("Posición", ["Todas"] + op["posiciones"], key="epos")
    e_liga = c2.selectbox("Liga", ["Todas"] + op["ligas"], key="eliga")
    e_pais = c3.selectbox("Nacionalidad", ["Todas"] + op["paises"], key="epais")
    # Clubes dependientes: solo los de la liga/nacionalidad elegidas
    clubs_op = get_clubs(e_liga if e_liga != "Todas" else None, e_pais if e_pais != "Todas" else None)
    e_club = c4.selectbox("Club", ["Todos"] + clubs_op)
    params = {}
    if e_pos != "Todas": params["posicion"] = e_pos
    if e_liga != "Todas": params["liga"] = e_liga
    if e_pais != "Todas": params["pais"] = e_pais
    if e_club != "Todos": params["club"] = e_club
    ev = requests.get(f"{API}/evolucion", params=params, timeout=10).json()
    serie, res = ev.get("serie", []), ev.get("resumen", {})
    if not serie:
        st.warning("No hay datos para ese filtro."); return
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Jugadores", f"{res['n_jugadores']:,}")
    r2.metric("Edad promedio", f"{res['edad_promedio']:.1f} años")
    r3.metric(f"Valor mediano ({res['anio_ultimo']})", eur(res["valor_mediano"]))
    mp = res.get("mov_proyectado_pct")
    r4.metric("Mov. proyectado 12m", f"{mp:+.1f}%" if mp is not None else "—",
              help="Crecimiento mediano proyectado por el modelo para este segmento")
    st.altair_chart(grafico_mercado(serie), use_container_width=True)
    st.caption("La mediana es el jugador típico; el promedio se va más arriba por los cracks. "
               "El salto general refleja la inflación del mercado de Transfermarkt.")


# ====================================================== PÁGINA: RANKING =======
def pagina_ranking():
    st.markdown("Elegí un **segmento de mercado** y mirá qué jugadores el modelo proyecta que más se "
                "**revalorizan** o **devalúan**. Tocá **Ver →** para ir a la ficha del jugador.")
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1, 1.4, 1.4])
    direccion = c1.radio("Dirección", ["suben", "bajan"],
                         format_func=lambda d: "📈 Suben" if d == "suben" else "📉 Bajan")
    tier_lbl = c2.selectbox("Segmento de valor", list(op["tiers"].values()))
    tier = [k for k, v in op["tiers"].items() if v == tier_lbl][0]
    r_pos = c3.selectbox("Posición", ["Todas"] + op["posiciones"], key="rpos")
    r_liga = c4.selectbox("Liga", ["Todas"] + op["ligas"], key="rliga")
    r_pais = c5.selectbox("Nacionalidad", ["Todas"] + op["paises"], key="rpais")
    params = {"direccion": direccion, "tier": tier, "limite": 15}
    if r_pos != "Todas": params["posicion"] = r_pos
    if r_liga != "Todas": params["liga"] = r_liga
    if r_pais != "Todas": params["pais"] = r_pais
    res = requests.get(f"{API}/ranking", params=params, timeout=10).json().get("resultados", [])
    if not res:
        st.warning("No hay jugadores con esos filtros en ese segmento."); return
    for i, x in enumerate(res, 1):
        col = VERDE if x["crecimiento_pct"] >= 0 else ROJO
        cc, cb = st.columns([6, 1])
        cc.markdown(f"<div class='rk' style='--c:{col}'><b style='font-family:Oswald;font-size:16px'>{i}. "
                    f"{x['nombre']}</b> <span style='color:#9fb3c8;font-size:13px'>{x['posicion']} · {x['edad']}a · "
                    f"{x['liga']} · {x['pais']}</span><br>{eur(x['valor_actual'])} → <b>{eur(x['valor_estimado'])}</b> "
                    f"<span style='color:{col};font-weight:800'>({x['crecimiento_pct']:+.0f}%)</span> "
                    f"<span style='color:#7e98b3;font-size:12px'>· rango {eur(x['p10'])}–{eur(x['p90'])}</span></div>",
                    unsafe_allow_html=True)
        cb.button("Ver →", key=f"ver_{x['player_id']}", on_click=ir_a, args=("consultar", x["player_id"]))


# ====================================================== PÁGINA: ASISTENTE =====
def pagina_asistente():
    st.markdown("Pedile al asistente que te recomiende jugadores en lenguaje natural. "
                "Ej: *“Tengo €85M y necesito un delantero, ¿cuál me recomendás?”* — te puede "
                "repreguntar y prioriza los que el modelo proyecta al alza.")
    st.session_state.setdefault("chat_msgs", [])
    if st.button("🗑️ Limpiar conversación"):
        st.session_state.chat_msgs = []

    def fichas_buttons(cands, key):
        # Botones "Ver ficha" para ir al detalle del jugador (como en Oportunidades)
        vistos, cols, i = set(), None, 0
        for c in cands:
            pid = c.get("player_id")
            if not pid or pid in vistos:
                continue
            vistos.add(pid)
            if i % 3 == 0:
                cols = st.columns(3)
            cols[i % 3].button(f"📊 {c['nombre']}", key=f"{key}_{pid}",
                               on_click=ir_a, args=("consultar", pid), use_container_width=True)
            i += 1
            if i >= 6:
                break

    for idx, m in enumerate(st.session_state.chat_msgs):
        with st.chat_message(m["role"], avatar="⚽" if m["role"] == "assistant" else "🧑"):
            st.markdown(m["content"])
            if m.get("candidatos"):
                st.caption("Ver la ficha completa de un jugador:")
                fichas_buttons(m["candidatos"], f"hist{idx}")

    prompt = st.chat_input("Ej: Tengo €85M y necesito un delantero. ¿Cuál me recomendás?")
    if prompt:
        st.session_state.chat_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="⚽"):
            with st.spinner("Pensando..."):
                try:
                    envio = [{"role": x["role"], "content": x["content"]} for x in st.session_state.chat_msgs]
                    resp = requests.post(f"{API}/chat", json={"mensajes": envio}, timeout=45).json()
                    texto = resp.get("respuesta", "No obtuve respuesta.")
                    cands = resp.get("candidatos", [])
                except Exception as e:
                    texto, cands = f"Error al consultar el asistente: {e}", []
            st.markdown(texto)
            if cands:
                st.caption("Ver la ficha completa de un jugador:")
                fichas_buttons(cands, "now")
        st.session_state.chat_msgs.append({"role": "assistant", "content": texto, "candidatos": cands})


PAG = {"consultar": pagina_consultar, "evolucion": pagina_evolucion,
       "ranking": pagina_ranking, "asistente": pagina_asistente}
PAG[st.session_state.page]()
