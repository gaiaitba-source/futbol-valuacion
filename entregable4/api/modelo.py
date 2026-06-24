"""
Capa de LÓGICA DEL MODELO (separada de la capa de servicio).

Carga los artefactos entrenados y expone funciones de negocio: buscar jugadores, predecir el
intervalo de valor a 12 meses (con fecha y confianza), trayectoria histórica REAL de un jugador,
evolución agregada del mercado (filtrable) y ranking de oportunidades.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

MODELOS = Path(__file__).resolve().parent.parent / 'modelos'
POSICIONES = {1: 'Arquero', 2: 'Defensor', 3: 'Mediocampista', 4: 'Delantero'}
TIERS = {
    'estrellas':    (30_000_000, 10**12, 'Estrellas (€30M+)'),
    'consolidados': (5_000_000, 30_000_000, 'Consolidados (€5–30M)'),
    'promesas':     (1_000_000, 5_000_000, 'Promesas (€1–5M)'),
    'todos':        (1_000_000, 10**12, 'Todos (€1M+)'),
}


class ModeloValuacion:
    def __init__(self, ruta: Path = MODELOS):
        self.meta = json.loads((ruta / 'metadata.json').read_text(encoding='utf-8'))
        self.features = self.meta['features']
        self.Q = float(self.meta['ajuste_conformal_Q'])
        self.cobertura = float(self.meta.get('metricas', {}).get('cobertura_intervalo', 0.80))
        self.qmodel = joblib.load(ruta / 'xgboost_cuantil.joblib')
        self.tabla = pd.read_csv(ruta / 'tabla_jugadores.csv')
        self.tabla['posicion'] = self.tabla['posicion_num'].map(POSICIONES).fillna('—')
        self.tabla['liga_display'] = self.tabla['liga_nombre'] + ' (' + self.tabla['liga_pais'] + ')'
        self.hist = pd.read_csv(ruta / 'historico.csv')
        self.hist['posicion'] = self.hist['posicion_num'].map(POSICIONES).fillna('—')
        self.hist['liga_display'] = self.hist['liga_nombre'] + ' (' + self.hist['liga_pais'] + ')'

    # --- Opciones de filtros ----------------------------------------------
    def opciones(self) -> dict:
        rec = self.tabla[self.tabla['anio'] >= 2024]
        ligas = [l for l in rec['liga_display'].value_counts().index if 'Otra' not in l][:40]
        paises = [p for p in rec['pais'].value_counts().head(40).index if p != '—']
        clubs = [c for c in rec['club'].value_counts().head(60).index if c != '—']
        return {'posiciones': ['Arquero', 'Defensor', 'Mediocampista', 'Delantero'],
                'ligas': sorted(ligas), 'paises': paises, 'clubs': sorted(clubs),
                'tiers': {k: v[2] for k, v in TIERS.items()}}

    def clubs_de(self, liga=None, pais=None):
        """Clubes (recientes) filtrados por liga y/o nacionalidad — para filtros dependientes."""
        df = self._filtrar(self.tabla[self.tabla['anio'] >= 2024], None, liga, pais, None)
        return [c for c in df['club'].value_counts().index if c != '—'][:80]

    def _filtrar(self, df, posicion=None, liga=None, pais=None, club=None):
        if posicion: df = df[df['posicion'] == posicion]
        if liga:     df = df[df['liga_display'] == liga]
        if pais:     df = df[df['pais'] == pais]
        if club:     df = df[df['club'] == club]
        return df

    # --- Búsqueda ----------------------------------------------------------
    def buscar(self, q, limite=20, posicion=None, liga=None, pais=None):
        if not q or not q.strip():
            return []
        m = self.tabla[self.tabla['name'].str.contains(q.strip(), case=False, na=False)]
        m = self._filtrar(m, posicion, liga, pais).sort_values('valor_actual', ascending=False).head(limite)
        return [{'player_id': int(r.player_id), 'nombre': r.name, 'posicion': r.posicion,
                 'edad': int(r.edad), 'anio': int(r.anio), 'liga': r.liga_display,
                 'pais': r.pais, 'club': r.club, 'valor_actual': float(r.valor_actual)} for r in m.itertuples()]

    # --- Predicción --------------------------------------------------------
    def predecir(self, player_id, overrides=None):
        fila = self.tabla[self.tabla['player_id'] == player_id]
        if fila.empty:
            raise KeyError(player_id)
        fila = fila.iloc[0]
        valor_actual = float(fila['valor_actual'])
        anio = int(fila['anio'])
        x = fila[self.features].astype(float).copy()
        permitidos = {'edad', 'contrato_restante', 'n_ap', 'goles_por_partido',
                      'asist_por_partido', 'minutos_promedio'}
        if overrides:
            for k, v in overrides.items():
                if k in permitidos and v is not None:
                    x[k] = float(v)
            if 'edad' in overrides:
                x['edad2'] = x['edad'] ** 2

        q10, q50, q90 = self.qmodel.predict(x.values.reshape(1, -1))[0]
        base = np.log1p(valor_actual)
        p10, p50, p90 = (max(v, 0.0) for v in sorted(
            [float(np.expm1(base + q10 - self.Q)), float(np.expm1(base + q50)),
             float(np.expm1(base + q90 + self.Q))]))
        crec = p50 / valor_actual - 1
        direccion = 'sube' if p10 > valor_actual else ('baja' if p90 < valor_actual else 'estable')
        return {'player_id': player_id, 'nombre': fila['name'], 'posicion': fila['posicion'],
                'edad': int(fila['edad']), 'liga': fila['liga_display'], 'pais': fila['pais'],
                'club': fila['club'], 'valor_actual': valor_actual, 'valor_estimado': p50,
                'intervalo': {'p10': p10, 'p90': p90},
                'crecimiento_pct': round(crec * 100, 1), 'direccion': direccion,
                'fecha_base': f"jun {anio}", 'fecha_objetivo': f"jun {anio + 1}",
                'confianza': round(self.cobertura, 3), 'es_simulacion': bool(overrides),
                'overrides_aplicados': {k: overrides[k] for k in (overrides or {}) if k in permitidos}}

    # --- Trayectoria histórica REAL de un jugador --------------------------
    def historico_jugador(self, player_id):
        h = self.hist[self.hist['player_id'] == player_id].sort_values('anio')
        if h.empty:
            raise KeyError(player_id)
        serie = [{'anio': int(r.anio), 'fecha': f"jun {int(r.anio)}", 'valor': float(r.valor),
                  'cambio_real_pct': (round(float(r.cambio_real_pct), 1)
                                      if pd.notna(r.cambio_real_pct) else None)}
                 for r in h.itertuples()]
        return {'player_id': player_id, 'nombre': h.iloc[-1]['name'], 'serie': serie}

    # --- Evolución agregada del mercado (filtrable) ------------------------
    def evolucion(self, posicion=None, liga=None, pais=None, club=None, player_id=None):
        h = self.hist
        if player_id:
            h = h[h['player_id'] == player_id]
        else:
            h = self._filtrar(h, posicion, liga, pais, club)
        h = h[h['anio'].between(2010, 2024)]   # 2025 es año incompleto → lo excluimos del agregado
        if h.empty:
            return {'serie': [], 'resumen': {}}
        agg = (h.groupby('anio')
               .agg(valor_mediano=('valor', 'median'), valor_promedio=('valor', 'mean'),
                    n=('player_id', 'nunique'), edad_prom=('edad', 'mean')).reset_index())
        serie = [{'anio': int(r.anio), 'valor_mediano': float(r.valor_mediano),
                  'valor_promedio': float(r.valor_promedio), 'n': int(r.n),
                  'edad_prom': round(float(r.edad_prom), 1)} for r in agg.itertuples()]
        ult = agg.iloc[-1]
        # Movimiento proyectado a 12m: mediana del crecimiento esperado (tabla de predicción)
        t = self.tabla[self.tabla['anio'] >= 2024]
        if player_id:
            t = t[t['player_id'] == player_id]
        else:
            t = self._filtrar(t, posicion, liga, pais, club)
        mov = float(np.median(t['crecimiento_pct'])) if len(t) else None
        resumen = {'anio_ultimo': int(ult['anio']), 'n_jugadores': int(ult['n']),
                   'edad_promedio': round(float(ult['edad_prom']), 1),
                   'valor_mediano': float(ult['valor_mediano']),
                   'mov_proyectado_pct': round(mov, 1) if mov is not None else None,
                   'n_proyeccion': int(len(t))}
        return {'serie': serie, 'resumen': resumen}

    # --- Ranking de oportunidades -----------------------------------------
    def ranking(self, direccion='suben', tier='consolidados', limite=12,
                posicion=None, liga=None, pais=None):
        lo, hi, _ = TIERS.get(tier, TIERS['todos'])
        df = self.tabla[(self.tabla['valor_actual'] >= lo) & (self.tabla['valor_actual'] < hi)
                        & (self.tabla['anio'] >= 2024)].copy()
        df = self._filtrar(df, posicion, liga, pais)
        df = df.sort_values('crecimiento_pct', ascending=(direccion == 'bajan')).head(limite)
        return [{'player_id': int(r.player_id), 'nombre': r.name, 'posicion': r.posicion,
                 'edad': int(r.edad), 'liga': r.liga_display, 'pais': r.pais,
                 'valor_actual': float(r.valor_actual), 'valor_estimado': float(r.p50),
                 'p10': float(r.p10), 'p90': float(r.p90),
                 'crecimiento_pct': round(float(r.crecimiento_pct), 1)} for r in df.itertuples()]
