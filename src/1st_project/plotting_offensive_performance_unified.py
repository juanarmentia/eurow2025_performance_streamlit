import json

import matplotlib.pyplot as plt
import pandas as pd
from click import style
from loguru import logger
from mplsoccer import Sbopen
from omegaconf import OmegaConf
import seaborn as sns
import numpy as np

import utils

config = OmegaConf.load('config.yml')

HIGHLIGHTED_PLAYER = config.player.full_name

parser = Sbopen()
# Competition ID (UEFA Women's Euro), Season 315 (2025)
matches = parser.match(53, 315)
match_ids = matches['match_id'].unique()

logger.info(f"Processing {len(match_ids)} matches...")

all_players_stats = {}

# Cargar CSV
df_xt = pd.read_csv('source/summary_players_xT_statsbomb_filtered.csv')[['player_name', 'xt']]

for match_id in match_ids:

    utils.get_match_data(parser, match_id, all_players_stats)



# all_players_stats_serializable = {}
# for player, stats in all_players_stats.items():
#     all_players_stats_serializable[player] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v
#                                               for k, v in stats.items()}
#
# with open('./source/all_players_stats_v1.json', 'w', encoding='utf-8') as f:
#     json.dump(all_players_stats_serializable, f, indent=4, ensure_ascii=False)


# with open('./source/all_players_stats.json', 'r', encoding='utf-8') as f:
#     all_players_stats = json.load(f)

# Convertir el diccionario a DataFrame
players_df = pd.DataFrame.from_dict(all_players_stats, orient='index')
players_df.reset_index(inplace=True)
players_df.rename(columns={'index': 'player_name'}, inplace=True)

players_df = pd.merge(players_df, df_xt, on='player_name', how='left')
players_df['xt'] = players_df['xt'].fillna(0)

# Calcular métricas por 90 minutos
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


players_df = players_df[players_df['minutes_total']>=90]
# players_df = players_df[(players_df['npxg_90']>0) & (players_df['xa_90']>0) & (players_df['goals_90']>0)]



# # Min-Max normalization
# for metric in config.metrics:
#     metric_name = metric.name
#     metric_min = players_df[metric_name].min()
#     metric_max = players_df[metric_name].max()
#
#     # Evitar división por cero
#     if metric_max - metric_min > 0:
#         players_df[f'{metric_name}_norm'] = (players_df[metric_name] - metric_min) / (metric_max - metric_min)
#     else:
#         players_df[f'{metric_name}_norm'] = 0.5  # Valor neutral si todos son iguales

# Z-score normalization
for metric in config.metrics:
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

# Calcular límites globales de z-score para todas las métricas
all_zscores = []
for metric in config.metrics:
    all_zscores.extend(players_df[f'{metric.name}_zscore'].values)

z_min_global = min(all_zscores)
z_max_global = max(all_zscores)
z_range_global = z_max_global - z_min_global

for metric in config.metrics:
    try:
        players_df[f'{metric.name}_Q'] = pd.qcut(players_df[metric.name], 4,
                                                 labels=[f'{metric.name}_Q1', f'{metric.name}_Q2',
                                                         f'{metric.name}_Q3', f'{metric.name}_Q4'],
                                                 duplicates='drop')
    except ValueError as e:
        # If qcut fails due to insufficient unique values, use cut instead
        logger.warning(f"qcut failed for {metric.name}, using cut instead: {e}")
        players_df[f'{metric.name}_Q'] = pd.cut(players_df[metric.name], 4,
                                                labels=[f'{metric.name}_Q1', f'{metric.name}_Q2',
                                                        f'{metric.name}_Q3', f'{metric.name}_Q4'],
                                                include_lowest=True)
    players_df[f'{metric.name}_Q'] = players_df[f'{metric.name}_Q'].cat.add_categories(HIGHLIGHTED_PLAYER)
    players_df.loc[players_df['player_name'] == HIGHLIGHTED_PLAYER, f'{metric.name}_Q'] = HIGHLIGHTED_PLAYER


fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12, 12), dpi=150)
fig.suptitle(f'Performance of {config.player.name}', fontsize=20, color='#15A9D6', y=0.94)
fig.text(x=0.5, y=0.88, ha='center', s="Sample of wingers of UEFA Women's Euro 2025 (>90min)", fontsize=12,
         style='italic', color='#3f3f3f')
fig.text(x=0.8, y=0.04, s='* Z-score per 90 minutes.', fontsize=9, fontweight='bold', color='#08306B', fontstyle='italic')
im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
newax_liga = fig.add_axes([0.82, 0.85, 0.13, 0.13], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')
im_player = plt.imread(f'./images/{HIGHLIGHTED_PLAYER.replace(" ", "_").lower()}.png')  # insert local path of the image.
newax_player = fig.add_axes([0.055, 0.84, 0.15, 0.15], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')

plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.93,
                    top=0.8,
                    wspace=0.2,
                    hspace=0.3)

# sns.set_palette("hls", 4)
cmap = 'Blues_r'


for metric in config.metrics:
    utils.create_horizontal_strip_plot_uni(axs, metric.ax[0]*4 + metric.ax[1], players_df, metric.name, metric.title,
                                           HIGHLIGHTED_PLAYER, z_min_global, z_max_global)
    # utils.create_swarm_plot_with_highlight(axs[metric.ax[0], metric.ax[1]], players_df, metric.name,
    #                                        f'{metric.name}_Q', cmap, metric.title, HIGHLIGHTED_PLAYER)


fig.savefig(f'./output/{HIGHLIGHTED_PLAYER.replace(" ", "_")}_performance.png', dpi=150)
