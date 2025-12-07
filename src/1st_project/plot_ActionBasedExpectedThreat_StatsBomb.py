# -*- coding: utf-8 -*-
"""
Action-based Expected Threat (xT) para StatsBomb
=================================================

Este script calcula el Expected Threat basado en acciones (action-based xT)
utilizando como entrada los ficheros JSON generados por
`plot_PossessionChain_StatsBomb.py` (uno por partido), que contienen las
"possession chains" y la geometría necesaria.

Flujo principal:
1) Cargar los JSON de `possession_chain/possession_chains_statsbomb_<match_id>.json`.
2) Para cada `match_id`, cargar eventos StatsBomb con `mplsoccer.Sbopen` para
   obtener `shot_statsbomb_xg` de los tiros y asignarlo a las cadenas que
   terminan en tiro (`shot_end == 1`). Propagar el valor de xG del tiro a todos
   los eventos de la cadena.
3) Binear las acciones en una rejilla del campo (por defecto 12x8 en sistema 120x80).
   Para cada bin origen→destino calcular:
   - Probabilidad de terminar en tiro (P(shot_end=1 | acción en bin)).
   - xG esperado dado tiro (E[xG | shot_end=1, acción en bin]).
4) Asignar a cada acción su `xT = shot_prob * xg_given_shot` según su bin.
5) Guardar resultados por acción y resúmenes por jugador/equipo.

Uso (por defecto procesa el partido demo 69301 si encuentra su JSON):
    python src/event_data/plot_ActionBasedExpectedThreat_StatsBomb.py \
        --input-dir possession_chain \
        --output-dir xT_output \
        --bins-x 12 --bins-y 8

Requisitos:
- Haber generado previamente los JSON de posesión con `plot_PossessionChain_StatsBomb.py`.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set

import numpy as np
import pandas as pd
from mplsoccer import Sbopen


# ============================
# Configuración por defecto
# ============================

PITCH_X = 120.0  # StatsBomb
PITCH_Y = 80.0   # StatsBomb


@dataclass
class GridSpec:
    bins_x: int = 12
    bins_y: int = 8

    def bin_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            np.linspace(0, PITCH_X, self.bins_x + 1),
            np.linspace(0, PITCH_Y, self.bins_y + 1),
        )


def load_possession_chain_files(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, 'possession_chains_statsbomb_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'No se encontraron ficheros con patrón {pattern}. Genere primero las cadenas de posesión.'
        )
    frames = []
    for f in files:
        frames.append(pd.read_json(f, dtype=False))
    df = pd.concat(frames, ignore_index=True)
    # Tipificar columnas clave por si vienen como float
    for col in ['match_id', 'team_id', 'player_id', 'possession_chain']:
        if col in df.columns:
            df[col] = df[col].astype('Int64')
    return df


def attach_shot_xg_to_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada partido, extrae `shot_statsbomb_xg` de eventos StatsBomb y
    lo asigna a las cadenas que terminan en tiro (`shot_end==1`).

    Emparejamos por (`match_id`, `minute`, `second`, `player_id`, `type_name=='Shot'`).
    Si no hay emparejamiento exacto, usamos un merge sólo por (`match_id`, `minute`, `player_id`).
    """
    out = df.copy()

    # Marcamos tiros en el DF de cadenas (debería existir si shot_end==1)
    is_shot = (out['type_name'] == 'Shot')

    # Cargar eventos por partido y crear tabla de tiros con xG
    parser = Sbopen()
    xg_rows = []
    for mid in sorted(out['match_id'].dropna().unique().tolist()):
        try:
            events, _, _, _ = parser.event(int(mid))
        except Exception:
            continue
        shots = events.loc[events['type_name'] == 'Shot',
                           ['match_id', 'minute', 'second', 'player_id', 'team_id', 'shot_statsbomb_xg']].copy()
        xg_rows.append(shots)
    if not xg_rows:
        # No pudimos cargar xG; devolvemos con xG de cadena = 0
        out['chain_xg'] = 0.0
        return out

    shots_all = pd.concat(xg_rows, ignore_index=True)

    # Merge exacto por minuto/segundo/jugador
    shots_all['second'] = shots_all['second'].round(0)
    out['second'] = out['second'].round(0)

    shots_key = ['match_id', 'minute', 'second', 'player_id']
    merged = out.merge(
        shots_all.rename(columns={'shot_statsbomb_xg': 'shot_xg'}),
        how='left', on=shots_key, suffixes=('', '_ev'))

    # Para filas no emparejadas pero que son tiros, intentar merge relajado por (match_id, minute, team)
    mask_need = merged['shot_xg'].isna() & (merged['type_name'] == 'Shot')
    if mask_need.any():
        relaxed = out.loc[mask_need, ['match_id', 'minute', 'team_id']].merge(
            shots_all[['match_id', 'minute', 'team_id', 'shot_statsbomb_xg']].rename(
                columns={'shot_statsbomb_xg': 'shot_xg'}),
            how='left', on=['match_id', 'minute', 'team_id']
        )['shot_xg'].values
        merged.loc[mask_need, 'shot_xg'] = relaxed

    # chain_xg: propagar el xG del tiro al resto de la cadena
    merged['chain_xg'] = 0.0
    # Tomar el xG del tiro por cadena (última acción debería ser el tiro)
    shot_in_chain = merged[merged['type_name'] == 'Shot'][['match_id', 'possession_chain', 'shot_xg']]
    shot_in_chain = shot_in_chain.dropna(subset=['shot_xg'])
    # Map a dict (match_id, chain) -> xg
    if not shot_in_chain.empty:
        chain_key = list(zip(shot_in_chain['match_id'].astype(int), shot_in_chain['possession_chain'].astype(int)))
        chain_xg_map: Dict[Tuple[int, int], float] = dict(zip(chain_key, shot_in_chain['shot_xg']))
        keys_series = list(zip(merged['match_id'].fillna(-1).astype(int), merged['possession_chain'].fillna(-1).astype(int)))
        merged['chain_xg'] = [chain_xg_map.get(k, 0.0) for k in keys_series]

    return merged


def build_bins(df: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    """Añade columnas de bin para inicio y fin de acción según rejilla StatsBomb."""
    x_edges, y_edges = grid.bin_edges()

    def to_bin(x, edges):
        # np.digitize devuelve índices 1..N, restamos 1 y cap al rango
        idx = np.digitize(x, edges, right=False) - 1
        return np.clip(idx, 0, len(edges) - 2)

    out = df.copy()
    out['bx0'] = to_bin(out['x0'].astype(float).values, x_edges)
    out['by0'] = to_bin(out['y0'].astype(float).values, y_edges)
    out['bx1'] = to_bin(out['x1'].astype(float).values, x_edges)
    out['by1'] = to_bin(out['y1'].astype(float).values, y_edges)

    # Índice combinado de par bin origen->destino
    out['bin_pair'] = (
        out['bx0'].astype(int) * grid.bins_y + out['by0'].astype(int)
    ).astype(int) * (grid.bins_x * grid.bins_y) + (
        out['bx1'].astype(int) * grid.bins_y + out['by1'].astype(int)
    ).astype(int)

    return out


def estimate_bin_models(df: pd.DataFrame, grid: GridSpec) -> pd.DataFrame:
    """Calcula por bin (origen->destino): shot_prob y xg_given_shot.

    - shot_prob: proporción de acciones en el bin cuya cadena terminó en tiro.
    - xg_given_shot: media de `chain_xg` para acciones en cadenas con tiro en ese bin.
    """
    g = df.groupby('bin_pair')

    # Shot probability
    shot_prob = g['shot_end'].mean().rename('shot_prob').fillna(0.0)

    # xG conditional on shot
    mask_shot = df['shot_end'] == 1
    xg_cond = df[mask_shot].groupby('bin_pair')['chain_xg'].mean().rename('xg_given_shot')

    model = pd.concat([shot_prob, xg_cond], axis=1)

    # Rellenos: si falta xg_given_shot para algún bin, usar media global de tiros
    global_xg = df.loc[df['shot_end'] == 1, 'chain_xg'].mean()
    if np.isnan(global_xg):
        global_xg = 0.1  # valor pequeño por defecto si no hay tiros
    model['xg_given_shot'] = model['xg_given_shot'].fillna(global_xg)

    # Si un bin no aparece en model, crear fila con ceros y global_xg al mapear más tarde.
    model = model.reset_index()
    model['global_xg_fallback'] = global_xg
    return model


def assign_xt(df: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Une el modelo por bin con las acciones y calcula xT = shot_prob * xg_given_shot."""
    out = df.merge(model[['bin_pair', 'shot_prob', 'xg_given_shot', 'global_xg_fallback']],
                   how='left', on='bin_pair')
    out['shot_prob'] = out['shot_prob'].fillna(0.0)
    # Si faltase xg_given_shot para algún bin desconocido
    out['xg_given_shot'] = out['xg_given_shot'].fillna(out['global_xg_fallback'])
    out['xT'] = out['shot_prob'] * out['xg_given_shot']
    return out.drop(columns=['global_xg_fallback'])


def summarize_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    by_player = df.groupby(['player_id', 'player_name'], dropna=False)['xT'].sum().reset_index()
    by_team = df.groupby(['team_id', 'team_name'], dropna=False)['xT'].sum().reset_index()
    by_player = by_player.sort_values('xT', ascending=False)
    by_team = by_team.sort_values('xT', ascending=False)
    return by_player, by_team


def _collect_primary_positions(match_ids: List[int]) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: player_id, player_name, primary_position.

    Usa `tactics` de StatsBomb (Sbopen) para contar apariciones por `position_name` y
    seleccionar la más frecuente por jugadora.
    """
    parser = Sbopen()
    rows = []
    for mid in sorted(set(int(m) for m in match_ids if pd.notna(m))):
        try:
            _, _, _, tactics = parser.event(mid)
        except Exception:
            continue
        # tactics suele contener columnas: player_id, player_name, position_name
        cols = [c for c in ['player_id', 'player_name', 'position_name'] if c in tactics.columns]
        if not cols or 'player_id' not in cols or 'position_name' not in cols:
            continue
        rows.append(tactics[cols].copy())
    if not rows:
        return pd.DataFrame(columns=['player_id', 'player_name', 'primary_position'])

    tac = pd.concat(rows, ignore_index=True)
    # Contar apariciones por jugador y posición
    counts = tac.groupby(['player_id', 'player_name', 'position_name']).size().reset_index(name='n')
    # Elegir posición más frecuente por jugador
    idx = counts.groupby('player_id')['n'].idxmax()
    primary = counts.loc[idx, ['player_id', 'player_name', 'position_name']].rename(
        columns={'position_name': 'primary_position'}
    ).reset_index(drop=True)
    return primary


def _apply_position_filter(
    actions_df: pd.DataFrame,
    positions: Optional[Set[str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (actions_filtered, players_filtered_summary) aplicando filtro por posiciones.

    Si `positions` es None o conjunto vacío, devuelve sin filtrar.
    """
    if not positions:
        by_player, _ = summarize_outputs(actions_df)
        return actions_df, by_player

    match_ids = actions_df['match_id'].dropna().unique().tolist()
    primary_positions = _collect_primary_positions(match_ids)
    if primary_positions.empty:
        # Si no se pudo obtener posiciones, devolver vacíos filtrados (no clasificables)
        filtered_actions = actions_df[[]].copy()
        filtered_players = pd.DataFrame(columns=['player_id', 'player_name', 'xT'])
        return filtered_actions, filtered_players

    actions_with_pos = actions_df.merge(primary_positions, how='left', on=['player_id', 'player_name'])
    filtered_actions = actions_with_pos[actions_with_pos['primary_position'].isin(list(positions))].copy()
    filtered_players = filtered_actions.groupby(['player_id', 'player_name'], dropna=False)['xT'].sum().reset_index()
    filtered_players = filtered_players.sort_values('xT', ascending=False)
    return filtered_actions, filtered_players


def main(input_dir: str, output_dir: str, bins_x: int, bins_y: int, filter_positions: Optional[List[str]] = None):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Cargar cadenas de posesión
    chains = load_possession_chain_files(input_dir)

    # 2) Adjuntar xG de tiros por cadena
    chains = attach_shot_xg_to_chains(chains)

    # 3) Binarizar acciones
    grid = GridSpec(bins_x=bins_x, bins_y=bins_y)
    chains = build_bins(chains, grid)

    # 4) Estimar modelo por bin y asignar xT
    model = estimate_bin_models(chains, grid)
    chains_xt = assign_xt(chains, model)

    # 5) Resúmenes
    by_player, by_team = summarize_outputs(chains_xt)

    # 6) Guardar (completo)
    actions_out = os.path.join(output_dir, 'actions_with_xT_statsbomb.json')
    model_out = os.path.join(output_dir, 'xt_model_bins_statsbomb.csv')
    players_out = os.path.join(output_dir, 'summary_players_xT_statsbomb.csv')
    teams_out = os.path.join(output_dir, 'summary_teams_xT_statsbomb.csv')

    chains_xt.to_json(actions_out, orient='records')
    model.to_csv(model_out, index=False)
    by_player.to_csv(players_out, index=False)
    by_team.to_csv(teams_out, index=False)

    print(f'Acciones con xT: {actions_out}')
    print(f'Modelo por bins: {model_out}')
    print(f'Resumen por jugadores: {players_out}')
    print(f'Resumen por equipos: {teams_out}')

    # 7) Filtrado por posiciones (opcional). Por defecto: extremos y mediocentros de banda.
    default_positions = {'Right Wing', 'Left Wing', 'Left Midfield', 'Right Midfield'}
    positions_set: Optional[Set[str]] = set(filter_positions) if filter_positions else default_positions

    filtered_actions, filtered_players = _apply_position_filter(chains_xt, positions_set)
    # Enriquecer acciones filtradas con columna primary_position si no estaba
    if 'primary_position' not in filtered_actions.columns:
        match_ids = chains_xt['match_id'].dropna().unique().tolist()
        primary_positions = _collect_primary_positions(match_ids)
        filtered_actions = filtered_actions.merge(primary_positions, how='left', on=['player_id', 'player_name'])

    actions_out_f = os.path.join(output_dir, 'actions_with_xT_statsbomb_filtered.json')
    players_out_f = os.path.join(output_dir, 'summary_players_xT_statsbomb_filtered.csv')

    filtered_actions.to_json(actions_out_f, orient='records')
    filtered_players.to_csv(players_out_f, index=False)
    print(f'Acciones filtradas por posición: {actions_out_f} (posiciones: {sorted(list(positions_set))})')
    print(f'Resumen jugadores filtrado: {players_out_f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Action-based xT para StatsBomb usando possession chains JSON')
    parser.add_argument('--input-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'possession_chain'),
                        help='Directorio con ficheros possession_chains_statsbomb_*.json')
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'xT_output'),
                        help='Directorio de salida para resultados xT')
    parser.add_argument('--bins-x', type=int, default=12, help='Número de bins en eje X (120)')
    parser.add_argument('--bins-y', type=int, default=8, help='Número de bins en eje Y (80)')
    parser.add_argument('--positions', type=str, nargs='*', default=['Right Wing', 'Left Wing', 'Left Midfield', 'Right Midfield'],
                        help='Lista de posiciones a filtrar (por nombre StatsBomb). Por defecto: bandas y mediocentros de banda.')
    args = parser.parse_args()

    main(input_dir=args.input_dir, output_dir=args.output_dir, bins_x=args.bins_x, bins_y=args.bins_y,
         filter_positions=args.positions)
