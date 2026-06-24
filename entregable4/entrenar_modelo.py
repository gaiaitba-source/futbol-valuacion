"""
=============================================================================
ENTREGABLE 4 — Entrenamiento y exportación del modelo desplegable
=============================================================================
Reproduce el pipeline del Entregable 3 (horizonte anclado, target de crecimiento,
sesgo de supervivencia corregido) y exporta los artefactos que consume la API:

  modelos/xgboost_cuantil.joblib   → modelo de cuantiles (genera el intervalo P10/P50/P90)
  modelos/modelo_puntual.joblib    → modelo puntual (XGBoost)
  modelos/metadata.json            → features, ajuste conformal, hiperparámetros, métricas
  modelos/tabla_jugadores.parquet  → una fila por jugador (último año) con sus features
                                     y la predicción ya calculada (para búsqueda y ranking)

Incluye el FIX del Entregable 3: el train se ordena por año antes del split temporal,
para que la validación temporal sea realmente temporal.

Uso:  python entrenar_modelo.py
"""
import warnings, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb

warnings.filterwarnings('ignore')

# --- Rutas -------------------------------------------------------------------
AQUI = Path(__file__).resolve().parent
CAND = [AQUI.parent / 'data' / 'raw', Path('../data/raw'), Path('data/raw')]
DATA = next((c for c in CAND if (c / 'players.csv').exists()), None)
if DATA is None:
    sys.exit('No encontré data/raw/. Ajustá la ruta.')
DATA = str(DATA) + '/'
SALIDA = AQUI / 'modelos'
SALIDA.mkdir(exist_ok=True)
print('Datos:', DATA)

# Hiperparámetros elegidos por RandomizedSearchCV en el Entregable 3 (CV temporal)
BEST_PARAMS = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.7)
DEVALUACION = 0.5          # caída asignada a los desaparecidos (elegida en E3)
LAST_SEASON = 2025

# --- Carga -------------------------------------------------------------------
players = pd.read_csv(DATA + 'players.csv')
app = pd.read_csv(DATA + 'appearances.csv', usecols=['player_id', 'game_id', 'player_club_id',
        'date', 'competition_id', 'goals', 'assists', 'minutes_played', 'yellow_cards'])
val = pd.read_csv(DATA + 'player_valuations.csv')
comp = pd.read_csv(DATA + 'competitions.csv', usecols=['competition_id', 'type', 'name', 'country_name'])
tr = pd.read_csv(DATA + 'transfers.csv', usecols=['player_id', 'transfer_date', 'transfer_fee'])
clubs = pd.read_csv(DATA + 'clubs.csv', usecols=['club_id', 'name']).rename(columns={'name': 'club'})
for d, c in [(app, 'date'), (val, 'date'), (tr, 'transfer_date')]:
    d[c] = pd.to_datetime(d[c], errors='coerce')
players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')
players['contract_year'] = pd.to_datetime(players['contract_expiration_date'], errors='coerce').dt.year
app['anio'] = app['date'].dt.year
val['anio'] = val['date'].dt.year
tr['anio'] = tr['transfer_date'].dt.year
print('Cargado.')

# --- Horizonte anclado al 30/6 (merge_asof) ----------------------------------
YEARS = list(range(2004, 2026))
pids = val['player_id'].unique()
grid = pd.MultiIndex.from_product([pids, YEARS], names=['player_id', 'T']).to_frame(index=False)
grid['cutoff'] = pd.to_datetime(dict(year=grid['T'], month=6, day=30))
grid = grid.sort_values('cutoff')
vs = val.dropna(subset=['market_value_in_eur']).sort_values('date')
va = pd.merge_asof(grid, vs[['player_id', 'date', 'market_value_in_eur',
        'player_club_domestic_competition_id']], left_on='cutoff', right_on='date',
        by='player_id', direction='backward').rename(
        columns={'market_value_in_eur': 'valor', 'player_club_domestic_competition_id': 'liga'})
va['fresh'] = (va['cutoff'] - va['date']).dt.days <= 460
va = va.sort_values(['player_id', 'T'])
va['valor_next'] = va.groupby('player_id')['valor'].shift(-1)
va['fresh_next'] = va.groupby('player_id')['fresh'].shift(-1)
va['pico_prev'] = va.groupby('player_id')['valor'].cummax()

# --- Rendimiento por año + rolling 2 años ------------------------------------
app = app.merge(comp[['competition_id', 'type']], on='competition_id', how='left')
app['es_inter'] = (app['type'] == 'international_cup').astype(int)
# Mapa liga (código tipo 'GB1') -> nombre y país, para los filtros de la interfaz
liga_map = comp[['competition_id', 'name', 'country_name']].rename(
    columns={'competition_id': 'liga', 'name': 'liga_nombre', 'country_name': 'liga_pais'})
st = app.groupby(['player_id', 'anio']).agg(
    n_ap=('game_id', 'count'), goles=('goals', 'sum'), asist=('assists', 'sum'),
    minutos=('minutes_played', 'sum'), amarillas=('yellow_cards', 'sum'),
    ap_inter=('es_inter', 'sum')).reset_index().sort_values(['player_id', 'anio'])
for c in ['n_ap', 'goles', 'asist', 'minutos']:
    st[c + '_roll2'] = st.groupby('player_id')[c].transform(lambda s: s.rolling(2, min_periods=1).mean())
st['goles_por_partido'] = st['goles'] / st['n_ap']
st['asist_por_partido'] = st['asist'] / st['n_ap']
st['minutos_promedio'] = st['minutos'] / st['n_ap']
st['jugo_internacional'] = (st['ap_inter'] > 0).astype(int)

club_principal = (app.groupby(['player_id', 'anio', 'player_club_id']).size().reset_index(name='n')
                  .sort_values('n').groupby(['player_id', 'anio']).tail(1)[['player_id', 'anio', 'player_club_id']])

# --- Contexto GLOBAL (consistente para entrenamiento y serving) --------------
mediana_liga = (val.dropna(subset=['player_club_domestic_competition_id', 'market_value_in_eur'])
    .groupby(['player_club_domestic_competition_id', 'anio'])['market_value_in_eur'].median()
    .reset_index().rename(columns={'market_value_in_eur': 'nivel_liga',
        'player_club_domestic_competition_id': 'liga', 'anio': 'anio_nivel'}))
infl = val.groupby('anio')['market_value_in_eur'].median()
infl = (infl / infl.get(2020, infl.median())).reset_index().rename(
    columns={'market_value_in_eur': 'nivel_mercado', 'anio': 'T'})
trans_anio = tr.groupby(['player_id', 'anio']).agg(fee=('transfer_fee', 'max')).reset_index()
trans_anio['transferido'] = 1
# nivel_club global: mediana del valor del plantel por (club, año), con todos los frescos
fresh_all = va[va['fresh'] & (va['valor'] > 0)].merge(club_principal,
    left_on=['player_id', 'T'], right_on=['player_id', 'anio'], how='left').drop(columns=['anio'])
nivel_club_tbl = fresh_all.groupby(['player_club_id', 'T'])['valor'].median().reset_index().rename(
    columns={'valor': 'nivel_club'})

ENC = {'Goalkeeper': 1, 'Defender': 2, 'Centre-Back': 2, 'Left-Back': 2, 'Right-Back': 2, 'Defence': 2,
       'Midfielder': 3, 'Midfield': 3, 'Central Midfield': 3, 'Defensive Midfield': 3,
       'Attacking Midfield': 3, 'Left Midfield': 3, 'Right Midfield': 3,
       'Forward': 4, 'Centre-Forward': 4, 'Left Winger': 4, 'Right Winger': 4,
       'Second Striker': 4, 'Attack': 4}

FEATURES = ['edad', 'edad2', 'es_arquero', 'es_defensor', 'es_mediocampista', 'es_delantero',
    'height_in_cm', 'pie_zurdo', 'pie_ambidiestro', 'contrato_restante', 'sin_dato_contrato',
    'n_ap', 'goles_por_partido', 'asist_por_partido', 'minutos_promedio', 'amarillas',
    'n_ap_roll2', 'goles_roll2', 'asist_roll2', 'minutos_roll2', 'jugo_internacional', 'ap_inter',
    'log_valor', 'dist_al_pico', 'flag_liga_top', 'nivel_liga', 'nivel_club', 'nivel_mercado',
    'transferido', 'ratio_fee']


def armar_features(muestra):
    """Pega todas las features a un conjunto de filas jugador-año (sin construir el target)."""
    d = muestra.merge(st, left_on=['player_id', 'T'], right_on=['player_id', 'anio'], how='left').drop(columns=['anio'])
    d = d.merge(club_principal, left_on=['player_id', 'T'], right_on=['player_id', 'anio'], how='left').drop(columns=['anio'])
    d = d.merge(nivel_club_tbl, on=['player_club_id', 'T'], how='left')
    d = d.merge(trans_anio, left_on=['player_id', 'T'], right_on=['player_id', 'anio'], how='left').drop(columns=['anio'])
    d['transferido'] = d['transferido'].fillna(0); d['fee'] = d['fee'].fillna(0)
    d['anio_nivel'] = d['T'] - 1
    d = d.merge(mediana_liga, on=['liga', 'anio_nivel'], how='left').drop(columns=['anio_nivel'])
    d = d.merge(infl, on='T', how='left')
    pf = players[['player_id', 'date_of_birth', 'position', 'sub_position', 'contract_year', 'height_in_cm', 'foot', 'name']]
    d = d.merge(pf, on='player_id', how='left')
    d['edad'] = d['T'] - d['date_of_birth'].dt.year
    d = d[(d['edad'] >= 15) & (d['edad'] <= 45)].copy()
    d['posicion_num'] = d['sub_position'].map(ENC)
    m = d['posicion_num'].isna(); d.loc[m, 'posicion_num'] = d.loc[m, 'position'].map(ENC)
    d['posicion_num'] = d['posicion_num'].fillna(3)
    for v, n in [(1, 'es_arquero'), (2, 'es_defensor'), (3, 'es_mediocampista'), (4, 'es_delantero')]:
        d[n] = (d['posicion_num'] == v).astype(int)
    d['contrato_restante'] = (d['contract_year'] - d['T']).clip(-1, 6)
    d['sin_dato_contrato'] = d['contract_year'].isna().astype(int)
    d['contrato_restante'] = d['contrato_restante'].fillna(1)
    d['height_in_cm'] = d['height_in_cm'].fillna(d['height_in_cm'].median())
    d['pie_zurdo'] = (d['foot'] == 'left').astype(int)
    d['pie_ambidiestro'] = (d['foot'] == 'both').astype(int)
    d['dist_al_pico'] = (d['valor'] / d['pico_prev']).clip(0, 1).fillna(1)
    d['ratio_fee'] = (d['fee'] / d['valor'].clip(lower=1)).clip(0, 10)
    d['flag_liga_top'] = d['liga'].isin(['GB1', 'ES1', 'IT1', 'DE1', 'FR1']).astype(int)
    d['edad2'] = d['edad'] ** 2
    d['log_valor'] = np.log1p(d['valor'])
    for c in ['n_ap', 'goles', 'asist', 'minutos', 'amarillas', 'ap_inter', 'n_ap_roll2',
              'goles_roll2', 'asist_roll2', 'minutos_roll2', 'goles_por_partido',
              'asist_por_partido', 'minutos_promedio', 'jugo_internacional']:
        if c in d: d[c] = d[c].fillna(0)
    return d


# --- Conjunto de entrenamiento (supervivientes + caídas) ---------------------
superv = va[va['fresh'] & (va['valor'] > 0) & (va['fresh_next'] == True)].copy()
superv['mv_next'] = superv['valor_next']; superv['desaparece'] = 0
desap = va[va['fresh'] & (va['valor'] > 0) & (va['fresh_next'] != True) & (va['T'] < LAST_SEASON)].copy()
desap['mv_next'] = (desap['valor'] * DEVALUACION).clip(lower=25000); desap['desaparece'] = 1
entren = armar_features(pd.concat([superv, desap], ignore_index=True))
entren['log_growth'] = np.log1p(entren['mv_next']) - np.log1p(entren['valor'])
entren = entren[np.isfinite(entren['log_growth'])]

# imputación de contexto con mediana de TRAIN
med_liga = entren[entren['T'] <= 2022]['nivel_liga'].median()
med_club = entren[entren['T'] <= 2022]['nivel_club'].median()
entren['nivel_liga'] = entren['nivel_liga'].fillna(med_liga)
entren['nivel_club'] = entren['nivel_club'].fillna(med_club)

# >>> FIX E3: ordenar por año antes del split temporal <<<
train = entren[entren['T'] <= 2022].sort_values('T').reset_index(drop=True)
test = entren[(entren['T'] >= 2023) & (entren['desaparece'] == 0)].copy()
X_train, y_train = train[FEATURES].fillna(0), train['log_growth']
X_test, y_test = test[FEATURES].fillna(0), test['log_growth']
print(f'Train {len(train)} | Test supervivientes {len(test)}')

# --- Entrenamiento -----------------------------------------------------------
modelo_puntual = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, tree_method='hist')
modelo_puntual.fit(X_train, y_train)

ALPHAS = np.array([0.1, 0.5, 0.9])
X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
qmodel = xgb.XGBRegressor(**BEST_PARAMS, objective='reg:quantileerror', quantile_alpha=ALPHAS,
                          random_state=42, tree_method='hist')
qmodel.fit(X_fit, y_fit)
q_cal = qmodel.predict(X_cal)
Q_CONF = float(np.quantile(np.maximum(q_cal[:, 0] - y_cal.values, y_cal.values - q_cal[:, 2]), 0.80))

# --- Métricas de control -----------------------------------------------------
val_act = test['valor'].values
ll_test = np.log1p(test['mv_next'].values)              # nivel real (escala log)
pl = np.log1p(val_act) + modelo_puntual.predict(X_test)
r2_lvl = r2_score(ll_test, pl)
mae = mean_absolute_error(test['mv_next'], np.clip(np.expm1(pl), 0, None))
q_te = qmodel.predict(X_test)
p10 = np.expm1(np.log1p(val_act) + q_te[:, 0] - Q_CONF)
p90 = np.expm1(np.log1p(val_act) + q_te[:, 2] + Q_CONF)
cobertura = float(((test['mv_next'] >= p10) & (test['mv_next'] <= p90)).mean())
print(f'MAE test €{mae/1e6:.2f}M | R2 nivel {r2_lvl:.4f} | cobertura intervalo {cobertura:.1%} | Q={Q_CONF:.4f}')

# --- Tabla de jugadores para servir (último año disponible por jugador) ------
ult = (va[va['fresh'] & (va['valor'] > 0)].sort_values('T')
       .groupby('player_id').tail(1).copy())
serv = armar_features(ult)
Xs = serv[FEATURES].fillna(0)
serv['nivel_liga'] = serv['nivel_liga'].fillna(med_liga)
serv['nivel_club'] = serv['nivel_club'].fillna(med_club)
Xs = serv[FEATURES].fillna(0)
qs = qmodel.predict(Xs)
lv = np.log1p(serv['valor'].values)
trio = np.clip(np.expm1(np.column_stack([lv + qs[:, 0] - Q_CONF, lv + qs[:, 1], lv + qs[:, 2] + Q_CONF])), 0, None)
trio = np.sort(trio, axis=1)   # forzar P10 <= P50 <= P90 (evita cruce de cuantiles)
serv['p10'], serv['p50'], serv['p90'] = trio[:, 0], trio[:, 1], trio[:, 2]
serv['crecimiento_pct'] = (serv['p50'] / serv['valor'] - 1) * 100
# Enriquecer con liga (nombre/país) y nacionalidad del jugador, para los filtros
serv = serv.merge(liga_map, on='liga', how='left')
serv = serv.merge(players[['player_id', 'country_of_citizenship']], on='player_id', how='left')
serv = serv.merge(clubs, left_on='player_club_id', right_on='club_id', how='left')
PRETTY = {'laliga': 'LaLiga', 'premier-league': 'Premier League', 'serie-a': 'Serie A',
          'ligue-1': 'Ligue 1', 'super-lig': 'Süper Lig', 'liga-portugal': 'Liga Portugal',
          'major-league-soccer': 'MLS', 'saudi-pro-league': 'Saudi Pro League'}
serv['liga_nombre'] = (serv['liga_nombre'].map(PRETTY)
                       .fillna(serv['liga_nombre'].str.replace('-', ' ').str.title()))
serv['liga_nombre'] = serv['liga_nombre'].fillna('Otra')
serv['liga_pais'] = serv['liga_pais'].fillna('—')
serv['country_of_citizenship'] = serv['country_of_citizenship'].fillna('—')
serv['club'] = serv['club'].fillna('—')
cols_meta = ['player_id', 'name', 'T', 'edad', 'posicion_num', 'liga', 'liga_nombre',
             'liga_pais', 'country_of_citizenship', 'club', 'valor', 'p10', 'p50', 'p90', 'crecimiento_pct']
tabla = serv[cols_meta + FEATURES].rename(
    columns={'T': 'anio', 'valor': 'valor_actual', 'country_of_citizenship': 'pais'})
tabla.to_csv(SALIDA / 'tabla_jugadores.csv', index=False)
print(f'Tabla de jugadores: {len(tabla)} filas')

# --- Histórico de mercado (valor REAL por año, para la pestaña de evolución) --
hist = va[va['fresh'] & (va['valor'] > 0)][['player_id', 'T', 'valor', 'valor_next', 'fresh_next', 'liga']].copy()
cp = club_principal.rename(columns={'anio': 'T'})
hist = hist.merge(cp, on=['player_id', 'T'], how='left')
hist = hist.merge(clubs, left_on='player_club_id', right_on='club_id', how='left')
perfil_e = players[['player_id', 'sub_position', 'position', 'date_of_birth', 'country_of_citizenship', 'name']].copy()
perfil_e['posicion_num'] = perfil_e['sub_position'].map(ENC)
mm = perfil_e['posicion_num'].isna()
perfil_e.loc[mm, 'posicion_num'] = perfil_e.loc[mm, 'position'].map(ENC)
perfil_e['posicion_num'] = perfil_e['posicion_num'].fillna(3)
perfil_e['birth_year'] = perfil_e['date_of_birth'].dt.year
hist = hist.merge(perfil_e[['player_id', 'posicion_num', 'country_of_citizenship', 'name', 'birth_year']],
                  on='player_id', how='left')
hist['edad'] = hist['T'] - hist['birth_year']
hist = hist.merge(liga_map, on='liga', how='left')
hist['liga_nombre'] = (hist['liga_nombre'].map(PRETTY)
                       .fillna(hist['liga_nombre'].str.replace('-', ' ').str.title())).fillna('Otra')
hist['liga_pais'] = hist['liga_pais'].fillna('—')
hist['country_of_citizenship'] = hist['country_of_citizenship'].fillna('—')
hist['club'] = hist['club'].fillna('—')
hist['cambio_real_pct'] = np.where(hist['fresh_next'] == True,
                                   (hist['valor_next'] / hist['valor'] - 1) * 100, np.nan)
hist_out = hist[(hist['T'] >= 2005) & (hist['edad'].between(15, 45))][
    ['player_id', 'name', 'T', 'edad', 'posicion_num', 'liga_nombre', 'liga_pais',
     'country_of_citizenship', 'club', 'valor', 'valor_next', 'fresh_next', 'cambio_real_pct']
].rename(columns={'T': 'anio', 'country_of_citizenship': 'pais'})
hist_out.to_csv(SALIDA / 'historico.csv', index=False)
print(f'Histórico de mercado: {len(hist_out)} filas')

# --- Persistencia ------------------------------------------------------------
joblib.dump(qmodel, SALIDA / 'xgboost_cuantil.joblib')
joblib.dump(modelo_puntual, SALIDA / 'modelo_puntual.joblib')
metadata = {
    'features': FEATURES, 'target': 'log_growth',
    'reconstruccion': 'valor_pred = expm1(log1p(valor_actual) + crecimiento_pred)',
    'cuantiles': ALPHAS.tolist(), 'ajuste_conformal_Q': Q_CONF,
    'hiperparametros': BEST_PARAMS, 'devaluacion_desaparecidos': DEVALUACION,
    'metricas': {'mae_euros': round(mae, 2), 'cobertura_intervalo': round(cobertura, 4)},
}
with open(SALIDA / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print('Artefactos guardados en', SALIDA)
for p in sorted(SALIDA.glob('*')): print('  -', p.name)
