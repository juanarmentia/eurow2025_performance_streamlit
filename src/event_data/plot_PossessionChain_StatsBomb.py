# -*- coding: utf-8 -*-
"""
Possession Chains (StatsBomb via mplsoccer)
==========================================

Script para cargar eventos de StatsBomb (Open Data) usando `mplsoccer.Sbopen`,
aislar cadenas de posesión (possession chains) y generar las variables necesarias
para calcular posteriormente el Expected Threat (xT) basado en acciones.

Salida: archivos JSON por partido en el directorio `possession_chain/` con, al menos,
las columnas: `match_id, possession_chain, possession_chain_team, team_name, type_name,
x0, y0, c0, x1, y1, c1, shot_end, xG`.

Notas:
- Coordenadas en sistema StatsBomb (120x80). `c` es la distancia al eje horizontal del medio del campo (y=40), en yardas.
- Para tiros, fijamos el punto final en (120, 40) para coherencia con el pipeline de xT.
- Esta implementación usa una aproximación estándar: una cadena termina cuando cambia el equipo en posesión
  o hay un evento de parada (falta, fuera de juego, saque de banda, fin de periodo, etc.).
"""

from __future__ import annotations

import os
import json
import pathlib
from typing import List

import numpy as np
import pandas as pd
from mplsoccer import Sbopen


STOP_TYPES = {
    # Eventos que detienen el juego/posesión
    'Foul Committed',
    'Offside',
    'Error',  # pérdidas claras
    'Ball Out',  # cuando esté disponible en el dataset
    'Referee Ball-Drop',
    'Half End',
    'Injury Stoppage',
}

NON_ONBALL_TYPES = {
    # Eventos administrativos/no relacionados con la conducción/pase/tiro
    'Starting XI', 'Half Start', 'Substitution', 'Tactical Shift', 'Player Off', 'Player On',
}


def _load_events(match_id: int) -> pd.DataFrame:
    parser = Sbopen()
    df, related, freeze, tactics = parser.event(match_id)

    # Normalizamos columnas utilizadas
    # Aseguramos las columnas de inicio/fin de acciones
    for col in ['end_x', 'end_y']:
        if col not in df.columns:
            df[col] = np.nan

    # Orden temporal por secuencia dentro del partido
    if 'index' in df.columns:
        df = df.sort_values(['index']).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def _is_stoppage(row: pd.Series) -> bool:
    t = row['type_name']
    if t in STOP_TYPES:
        return True
    # Pass fallido que sale fuera o fuera de juego: cuenta como fin de cadena
    if t == 'Pass':
        outcome = row.get('pass_outcome_name') or row.get('outcome_name')
        if outcome in {'Out', 'Unknown', 'Incomplete', 'Pass Offside'}:
            return True
    # Duelos: si el resultado es perdido para el equipo en posesión, cortamos cadena
    if t == 'Duel':
        outcome = row.get('duel_outcome_name') or row.get('outcome_name')
        if outcome in {'Lost', 'Lost Out', 'Lost In Play'}:
            return True
    # Tiro siempre termina cadena
    if t == 'Shot':
        return True
    return False


def _is_onball(row: pd.Series) -> bool:
    t = row['type_name']
    if t in NON_ONBALL_TYPES:
        return False
    return True


def isolate_chains_sb(events: pd.DataFrame) -> pd.DataFrame:
    """Asigna `possession_chain` y `possession_chain_team` por partido.

    Reglas principales:
    - Se ignoran eventos no relacionados con la posesión (alineaciones, cambios, etc.).
    - Se agrupa una cadena mientras el equipo no cambie y no aparezca un evento de parada.
    - Un tiro cierra la cadena.
    """
    df = events.copy()

    # Nos quedamos con eventos relevantes para determinar la posesión
    mask_onball = df.apply(_is_onball, axis=1)
    df = df[mask_onball].copy().reset_index(drop=True)

    chain_id = 0
    current_team = None

    df['possession_chain'] = -1
    df['possession_chain_team'] = -1

    for i, row in df.iterrows():
        team_id = row['team_id']
        if current_team is None:
            current_team = team_id
        # Si cambia el equipo, nueva cadena
        if team_id != current_team:
            chain_id += 1
            current_team = team_id

        df.at[i, 'possession_chain'] = chain_id
        df.at[i, 'possession_chain_team'] = current_team

        # Si el evento detiene el juego, avanzamos a la siguiente cadena
        if _is_stoppage(row):
            chain_id += 1
            current_team = None

    return df


def add_geometry_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula x0, y0, c0, x1, y1, c1 y marca `shot_end`.

    - c es distancia vertical al centro del campo (y=40) en el sistema 120x80.
    - Para tiros, forzamos el destino (x1, y1) = (120, 40).
    """
    out = df.copy()

    # Coords de inicio
    out['x0'] = out['x']
    out['y0'] = out['y']
    out['c0'] = (out['y'] - 40).abs()

    # Coords de fin
    out['x1'] = out['end_x']
    out['y1'] = out['end_y']

    # Si falta fin para pases/duelos, copia inicio
    for col in ['x1', 'y1']:
        out[col] = out[col].fillna(out[col[0] + '0'])

    # Estandariza tiros
    shot_mask = out['type_name'] == 'Shot'
    out.loc[shot_mask, 'x1'] = 120.0
    out.loc[shot_mask, 'y1'] = 40.0

    out['c1'] = (out['y1'] - 40).abs()

    # shot_end por cadena: marcamos 1 a toda la cadena que termina con tiro
    out['shot_end'] = 0
    for chain_id, chain_df in out.groupby('possession_chain'):
        if not chain_df.empty and (chain_df.iloc[-1]['type_name'] == 'Shot'):
            out.loc[out['possession_chain'] == chain_id, 'shot_end'] = 1

    # xG placeholder a 0. El cálculo real se realiza en el script de xT posterior
    out['xG'] = 0.0

    return out


def filter_possession_team_only(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva sólo los eventos del equipo en posesión dentro de cada cadena."""
    keep_idx: List[int] = []
    for chain_id, chain in df.groupby('possession_chain'):
        if chain.empty:
            continue
        team_mode = chain['team_id'].mode().iloc[0]
        keep_idx.extend(chain[chain['team_id'] == team_mode].index.tolist())
    return df.loc[sorted(keep_idx)].copy()


def build_possession_chains_for_match(match_id: int) -> pd.DataFrame:
    events = _load_events(match_id)

    # Aislar cadenas
    chains = isolate_chains_sb(events)

    # Añadir geometría y etiquetas
    chains = add_geometry_and_labels(chains)

    # Mantener sólo eventos del equipo en posesión en cada cadena
    chains = filter_possession_team_only(chains)

    # Columnas mínimas para el pipeline de xT
    cols = [
        'match_id', 'minute', 'second', 'team_id', 'team_name', 'player_id', 'player_name',
        'type_name', 'possession_chain', 'possession_chain_team',
        'x0', 'y0', 'c0', 'x1', 'y1', 'c1', 'shot_end', 'xG'
    ]
    for c in cols:
        if c not in chains.columns:
            chains[c] = np.nan

    return chains[cols].copy()


def save_chains(df: pd.DataFrame, match_id: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'possession_chains_statsbomb_{match_id}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json.loads(df.to_json(orient='records')), f, ensure_ascii=False)
    return out_path


def main(match_ids: List[int] | None = None):
    """Ejecuta el pipeline para una lista de partidos.

    Si `match_ids` es None, procesa un partido de ejemplo (69301: ENG vs SWE - demo).
    """
    if not match_ids:
        match_ids = [69301]

    out_dir = os.path.join(str(pathlib.Path().resolve().parents[0]), 'possession_chain')

    for mid in match_ids:
        chains = build_possession_chains_for_match(mid)
        out_path = save_chains(chains, mid, out_dir)
        print(f'Guardado: {out_path}  (eventos: {len(chains)})')


if __name__ == '__main__':
    parser = Sbopen()
    # Competition ID (UEFA Women's Euro), Season 315 (2025)
    matches = parser.match(53, 315)
    match_ids = matches['match_id'].unique()
    # Puedes editar la lista para procesar múltiples partidos de una competición
    # main()
    main(match_ids.tolist())
