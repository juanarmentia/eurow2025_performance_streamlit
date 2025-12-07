# -*- coding: utf-8 -*-
"""
Player xT Heat Map from possession chain files (StatsBomb)
==========================================================

This script builds a heat map for a selected player using action-based
Expected Threat (xT) computed from possession-chain JSON files produced by
`plot_PossessionChain_StatsBomb.py`.

Workflow:
1) Load JSON files from `--input-dir` (default: possession_chain/).
2) Compute action-based xT using helper functions from
   `plot_ActionBasedExpectedThreat_StatsBomb.py`.
3) Filter actions for the requested `--player` (exact or case-insensitive substring).
4) Produce a 2D heat map of action start locations weighted by xT (or by count).

Usage example:
    python src/event_data/plot_PlayerxTHeatMap_StatsBomb.py \
        --input-dir possession_chain \
        --player "Mariona Caldentey" \
        --bins-x 12 --bins-y 8 \
        --output xT_output/Mariona_Caldentey_xT_heatmap.png

Requirements:
- possession_chain JSON files must exist.
- mplsoccer and matplotlib installed.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from omegaconf import OmegaConf

# Adding the colorbar without affecting the pitch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Reuse the xT pipeline utilities
from plot_ActionBasedExpectedThreat_StatsBomb import (
    GridSpec,
    load_possession_chain_files,
    attach_shot_xg_to_chains,
    build_bins,
    estimate_bin_models,
    assign_xt,
)

config = OmegaConf.load('config.yml')

def _compute_actions_with_xt(input_dir: str, bins_x: int, bins_y: int) -> pd.DataFrame:
    chains = load_possession_chain_files(input_dir)

    # Basic expected columns in chains JSONs
    required_cols = {'x0', 'y0', 'x1', 'y1', 'type_name', 'match_id', 'team_id', 'player_id', 'player_name'}
    missing = required_cols - set(chains.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en JSON: {sorted(missing)}")

    # shot_end should exist; if not, infer from 'chain_end' and 'type_name'
    if 'shot_end' not in chains.columns:
        chains = chains.copy()
        chains['shot_end'] = (chains['type_name'] == 'Shot').astype(int)

    # Attach xG of the ending shot to the whole chain
    with_xg = attach_shot_xg_to_chains(chains)

    # Bin actions and estimate bin model
    grid = GridSpec(bins_x=bins_x, bins_y=bins_y)
    binned = build_bins(with_xg, grid)
    model = estimate_bin_models(binned, grid)

    # Assign xT per action
    actions_xt = assign_xt(binned, model)
    return actions_xt


def _select_player(df: pd.DataFrame, player: str) -> pd.DataFrame:
    # Exact first, then case-insensitive contains as fallback
    exact = df['player_name'].fillna('').astype(str) == player
    if exact.any():
        return df.loc[exact].copy()
    mask = df['player_name'].fillna('').str.contains(player, case=False, regex=False)
    return df.loc[mask].copy()


def _pitch_histogram(x: np.ndarray, y: np.ndarray, bins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # StatsBomb pitch is 120x80 with origin (0,0) at bottom-left in mplsoccer
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 120], [0, 80]])
    return H.T, xedges, yedges  # transpose to align with pitch orientation


def _pitch_histogram_weighted(x: np.ndarray, y: np.ndarray, w: np.ndarray, bins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 120], [0, 80]], weights=w)
    return H.T, xedges, yedges


def plot_player_xt_heatmap(df_player: pd.DataFrame, player_label: str, bins_x: int, bins_y: int,
                           weight: str = 'xT', output: Optional[str] = None) -> None:
    # Prepare data
    x = df_player['x0'].astype(float).to_numpy()
    y = df_player['y0'].astype(float).to_numpy()

    if weight == 'count':
        H, xedges, yedges = _pitch_histogram(x, y, bins=(bins_x, bins_y))
        cmap = 'Blues'
        legend_label = 'Action count'
    else:
        w = df_player['xT'].fillna(0.0).to_numpy()
        H, xedges, yedges = _pitch_histogram_weighted(x, y, w, bins=(bins_x, bins_y))
        cmap = 'Reds'
        legend_label = 'xT sum'

    # Plot
    pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='grey', line_zorder=2)

    # fig, ax = pitch.draw(figsize=(9, 7))
    # plt.subplots_adjust(left=0.15, right=0.80, top=0.75, bottom=0.20)
    fig = plt.figure(figsize=(12, 8))

    # --- MANUAL AXES CREATION FOR SMALLER PITCH ---
    # Defines the axes position: [left, bottom, width, height] (normalized to 0-1)
    # Pitch will be placed in the center, taking up 50% of the figure width and height.
    ax = fig.add_axes([0.1, 0.1, 0.77, 0.78])

    # Now, draw the pitch into the manually sized axes
    pitch.draw(ax=ax)

    # Plot the kdeplot with reduced opacity for visibility
    kde = pitch.kdeplot(
        df_player['x0'],
        df_player['y0'],
        ax=ax,
        fill=True,
        thresh=0,
        n_levels=100,
        cmap='RdPu'  # Adjust transparency for better pitch visibility
    )

    # Create a normalization and colormap for the colorbar
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Adjust the range if needed
    sm = cm.ScalarMappable(cmap='RdPu', norm=norm)
    sm.set_array([])

    # Add the color bar beside the pitch
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.05)
    cbar.set_label('xT sum', fontsize=12)

    fig.suptitle(f"Accumulation of xThreat of {config.player.name}", fontsize=20, color='#15A9D6', y=0.94)
    im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
    newax_liga = fig.add_axes([0.75, 0.85, 0.13, 0.13], anchor='NE', zorder=1)
    newax_liga.imshow(im_liga)
    newax_liga.axis('off')
    im_player = plt.imread(
        f'./images/{config.player.full_name.replace(" ", "_").lower()}.png')  # insert local path of the image.
    newax_player = fig.add_axes([0.07, 0.85, 0.14, 0.14], anchor='NE', zorder=1)
    newax_player.imshow(im_player)
    newax_player.axis('off')

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches='tight')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Player xT heat map from possession chains (StatsBomb)')
    parser.add_argument('--weight', choices=['xT', 'count'], default='xT', help='Heat weight: xT (sum) or count')

    args = parser.parse_args()

    actions_xt = _compute_actions_with_xt('../possession_chain', 12, 8)

    team_mask = actions_xt['team_name'].fillna('').str.contains(config.player.team, case=False, regex=False)
    actions_xt = actions_xt.loc[team_mask].copy()

    player_df = _select_player(actions_xt, config.player.full_name)
    if player_df.empty:
        raise SystemExit(f"No se encontraron acciones para el jugador '{config.player.full_name}'.")

    plot_player_xt_heatmap(player_df, config.player.full_name, 12, 8, weight=args.weight, output=f'output/{config.player.name}_xt.png')


if __name__ == '__main__':
    main()
