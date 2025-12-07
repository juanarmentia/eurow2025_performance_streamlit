import json

import numpy as np
import streamlit as st
import pandas as pd
from typing import List, Tuple
from collections import Counter

from matplotlib import pyplot as plt
from mplsoccer import Sbopen

from omegaconf import OmegaConf

import utils

config = OmegaConf.load('src/1st_project/config.yml')

metrics_dict = {metric.name: metric.title for metric in config.metrics}

# ---------------------------
# Data loading helpers
# ---------------------------

@st.cache_data(show_spinner=True)
def load_all_players_stats() -> pd.DataFrame:
    # Collecting all stats
    df_xt = pd.read_csv('src/1st_project/source/summary_players_xT_statsbomb_filtered.csv')[['player_name', 'xt']]

    with open('src/1st_project/source/all_players_stats_streamlit_v1.json', 'r', encoding='utf-8') as f:
        all_players_stats = json.load(f)

    # Convertir el diccionario a DataFrame
    players_df = pd.DataFrame.from_dict(all_players_stats, orient='index')
    players_df.reset_index(inplace=True)
    players_df.rename(columns={'index': 'player_name'}, inplace=True)

    players_df = pd.merge(players_df, df_xt, on='player_name', how='left')
    players_df['xt'] = players_df['xt'].fillna(0)

    # Calcular métricas por 90 minutos
    players_df['position'] = players_df['position'].apply(
        lambda x: Counter(x).most_common(1)[0][0] if x and len(x) > 0 else 'Unknown'
    )
    players_df['minutes_total'] = players_df['minutes']
    players_df['minutes_90'] = players_df['minutes'] / 90
    players_df['goals_90'] = players_df['goals'] / players_df['minutes_90']
    players_df['shots_90'] = players_df['shots'] / players_df['minutes_90']
    players_df['npxg_90'] = players_df['npxg'] / players_df['minutes_90']
    players_df['assists_90'] = players_df['assists'] / players_df['minutes_90']
    players_df['xa_90'] = players_df['xa'] / players_df['minutes_90']
    players_df['key_passes_90'] = players_df['key_passes'] / players_df['minutes_90']
    players_df['progr_passes_90'] = players_df['progr_passes'] / players_df['minutes_90']
    players_df['progr_passes_rec_90'] = players_df['progr_passes_rec'] / players_df['minutes_90']
    players_df['dribbles_90'] = players_df['dribbles'] / players_df['minutes_90']
    players_df['carries_90'] = players_df['carries'] / players_df['minutes_90']
    players_df['progr_carries_90'] = players_df['progr_carries'] / players_df['minutes_90']
    players_df['xt_90'] = players_df['xt'] / players_df['minutes_90']
    players_df['interceptions_90'] = players_df['interceptions'] / players_df['minutes_90']
    players_df['ballrecoveries_90'] = players_df['ball_recoveries'] / players_df['minutes_90']
    players_df['tackles_90'] = players_df['total_tackles'] / players_df['minutes_90']
    players_df['successful_tackles_90'] = players_df['successful_tackles'] / players_df['minutes_90']
    players_df['successful_tackles_perc'] = players_df['successful_tackles'] / players_df['total_tackles']

    players_df = players_df[players_df['minutes_total'] >= 90]

    # Z-score normalization
    for metric in (config.metrics + config.metrics_def):
        metric_name = metric.name
        metric_mean = players_df[metric_name].mean()
        metric_std = players_df[metric_name].std()

        # Calcular z-score
        if metric_std > 0:
            players_df[f'{metric_name}_zscore'] = (players_df[metric_name] - metric_mean) / metric_std
        else:
            players_df[f'{metric_name}_zscore'] = 0  # Valor neutral si no hay variación

        # Calculate percentile for each player
        players_df[f'{metric_name}_percentile'] = players_df[metric_name].rank(pct=True) * 100

    return players_df

@st.cache_data(show_spinner=True)
def load_matches() -> pd.DataFrame:
    parser = Sbopen()
    # UEFA Women's Euro 2025: competition_id=53, season_id=315
    matches = parser.match(53, 315)
    return matches



@st.cache_data(show_spinner=True)
def load_events(match_id: int) -> pd.DataFrame:
    parser = Sbopen()
    events, related, freeze, tactics = parser.event(match_id)
    return events


# @st.cache_data(show_spinner=True)
# def load_all_events_for_teams(team_names: Tuple[str, ...]) -> pd.DataFrame:
#     matches = load_matches()
#     team_matches = matches[(matches['home_team_name'].isin(team_names)) | (matches['away_team_name'].isin(team_names))]
#     dfs = []
#     for _, row in team_matches.iterrows():
#         try:
#             ev = load_events(int(row['match_id']))
#             dfs.append(ev)
#         except Exception:
#             # Skip matches that fail to load
#             continue
#     if dfs:
#         return pd.concat(dfs, ignore_index=True)
#     return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_all_events_for_team_players(team_name: str) -> pd.DataFrame:
    matches = load_matches()
    team_matches = matches[(matches['home_team_name'] == team_name) | (matches['away_team_name'] == team_name)]
    dfs = []
    for _, row in team_matches.iterrows():
        try:
            ev = load_events(int(row['match_id']))
            ev = ev[ev['team_name'] == team_name]
            dfs.append(ev)
        except Exception:
            # Skip matches that fail to load
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ---------------------------
# Metric computation
# ---------------------------

def compute_player_metrics(events: pd.DataFrame, player_name: str) -> dict:
    if events.empty:
        return {
            'Shots': 0,
            'Goals': 0,
            'xG': 0.0,
            'Shot Assists': 0,
            'Touches': 0,
        }

    df = events[events['player_name'] == player_name].copy()
    shots = df[df['type_name'] == 'Shot']
    goals = shots[shots['outcome_name'] == 'Goal'] if 'outcome_name' in shots.columns else pd.DataFrame()
    # StatsBomb xG column name
    xg_col = 'shot_statsbomb_xg' if 'shot_statsbomb_xg' in shots.columns else None
    xg = float(shots[xg_col].sum()) if xg_col else 0.0

    # Shot assists are passes that directly precede a shot
    pass_df = df[df['type_name'] == 'Pass']
    shot_assists = 0
    if 'shot_assist' in pass_df.columns:
        shot_assists = int(pass_df['shot_assist'].fillna(False).astype(bool).sum())

    touches = int(df.shape[0])  # number of events by the player as a simple proxy

    return {
        'Shots': int(shots.shape[0]),
        'Goals': int(goals.shape[0]) if not goals.empty else 0,
        'xG': round(xg, 2),
        'Shot Assists': shot_assists,
        'Touches': touches,
    }


# ---------------------------
# Streamlit UI
# ---------------------------

col_header1, col_header2 = st.columns([0.95, 0.05])

st.set_page_config(page_title="EURO 2025 (W) – StatsBomb Open Data", layout="wide")
with col_header1:
    st.title("UEFA Women's Euro 2025 - Players performance")
    st.caption("Data source: StatsBomb Open Data")
with col_header2:
    st.image('src/1st_project/images/euro.png', width='content')

matches_df = load_matches()

# tab_player, tab_comparison = st.tabs(["Player Performance", "Players Comparison"])

all_metrics_off_list = []
for metric in config.metrics:
    all_metrics_off_list.append(metric.name)

all_metrics_def_list = []
for metric in config.metrics_def:
    all_metrics_def_list.append(metric.name)

players_df = load_all_players_stats()


# with tab_player:
# st.subheader("Players Performance")
col1, col2 = st.columns([0.2, 0.8], gap='large')
with col1:
    all_teams = sorted(pd.unique(pd.concat([matches_df['home_team_name'], matches_df['away_team_name']]).dropna()))
    team = st.selectbox("Select team", options=all_teams)
    events_team = load_all_events_for_team_players(team) if team else pd.DataFrame()
    if events_team.empty:
        st.info("No events available for the selected team yet.")
    else:
        players = sorted(events_team['player_name'].dropna().unique().tolist())
        selected_player = st.selectbox("Player", options=players, index=0 if players else None)

    # Inicializar session state si no existe
    if 'selected_positions' not in st.session_state:
        st.session_state.selected_positions = list(config.positions)[:3]

    # Cuando se selecciona un jugador, actualizar las posiciones
    if selected_player:
        player_position = players_df[players_df['player_name'] == selected_player]['position'].iloc[0]

        # Añadir la posición si no está ya en la lista
        if player_position in list(config.positions):
            if player_position not in st.session_state.selected_positions:
                st.session_state.selected_positions.append(player_position)

    positions = st.multiselect(
        "Select positions to compare:",
        options=list(config.positions),
        default=st.session_state.selected_positions,
    )

    # IMPORTANTE: Actualizar el session state con lo que el usuario selecciona manualmente
    st.session_state.selected_positions = positions

    # Actualizar session state con la selección actual del usuario
    if positions:
        st.session_state.selected_positions = positions

    metrics_off = st.multiselect(
        "Select Offensive Metrics:",
        all_metrics_off_list,
        default=all_metrics_off_list[:3],
    )

    metrics_def = st.multiselect(
        "Select Defensive Metrics:",
        all_metrics_def_list,
        default=all_metrics_def_list[:3],
    )

with col2:
    if selected_player and (metrics_off or metrics_def):
        position = players_df[players_df['player_name'] == selected_player]['position'].iloc[0]
        specific_position_df = players_df[players_df['position'].isin(positions)].copy()
        filtered_df = pd.concat([players_df[players_df['player_name'] == selected_player], specific_position_df], ignore_index=True)
        # Calcular límites globales de z-score para todas las métricas
        all_zscores = []
        metrics = metrics_off + metrics_def
        for metric in metrics:
            all_zscores.extend(filtered_df[f'{metric}_zscore'].values)

        z_min_global = min(all_zscores)
        z_max_global = max(all_zscores)
        z_range_global = z_max_global - z_min_global

        for metric in metrics:
            try:
                filtered_df[f'{metric}_Q'] = pd.qcut(filtered_df[metric], 4,
                                                         labels=[f'{metric}_Q1', f'{metric}_Q2',
                                                                 f'{metric}_Q3', f'{metric}_Q4'],
                                                         duplicates='drop')
            except ValueError as e:
                # If qcut fails due to insufficient unique values, use cut instead
                filtered_df[f'{metric}_Q'] = pd.cut(filtered_df[metric], 4,
                                                        labels=[f'{metric}_Q1', f'{metric}_Q2',
                                                                f'{metric}_Q3', f'{metric}_Q4'],
                                                        include_lowest=True)
            filtered_df[f'{metric}_Q'] = filtered_df[f'{metric}_Q'].cat.add_categories(selected_player)
            filtered_df.loc[filtered_df['player_name'] == selected_player, f'{metric}_Q'] = selected_player

        fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(8, len(metrics)*0.8), dpi=150)
        fig.suptitle(f'{selected_player} ({position})', fontsize=14, color='#15A9D6', y=0.94)
        fig.text(x=0.8, y=0, s='* Z-score per 90 minutes.', fontsize=8, fontweight='bold', color='#08306B',
                 fontstyle='italic')

        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.93,
                            top=0.8,
                            wspace=0.2,
                            hspace=0.3)

        # sns.set_palette("hls", 4)
        cmap = 'Blues_r'

        for idx, metric_name in enumerate(metrics):
            title = None
            for metric in config.metrics:
                if metric.name == metric_name:
                    title = metric.title
                    break
            if title is None:
                for metric in config.metrics_def:
                    if metric.name == metric_name:
                        title = metric.title
                        break
            utils.create_horizontal_strip_plot_streamlit(axs, idx, filtered_df, metric_name, len(metrics),
                                                   title, selected_player, z_min_global, z_max_global)

        st.pyplot(fig)


