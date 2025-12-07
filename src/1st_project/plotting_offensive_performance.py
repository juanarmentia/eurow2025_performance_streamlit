import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from mplsoccer import Sbopen
from omegaconf import OmegaConf
import seaborn as sns

import utils

config = OmegaConf.load('config.yml')

HIGHLIGHTED_PLAYER = config.player.full_name

parser = Sbopen()
# Competition ID (UEFA Women's Euro), Season 315 (2025)
matches = parser.match(53, 315)
match_ids = matches['match_id'].unique()

logger.info(f"Processing {len(match_ids)} matches...")

all_players_stats = {}

for match_id in match_ids:

    utils.get_match_data(parser, match_id, all_players_stats)



# all_players_stats_serializable = {}
# for player, stats in all_players_stats.items():
#     all_players_stats_serializable[player] = {k: int(v) if isinstance(v, (np.integer, np.int64)) else float(v) if isinstance(v, (np.floating, np.float64)) else v
#                                               for k, v in stats.items()}
#
# with open('./source/all_players_stats.json', 'w', encoding='utf-8') as f:
#     json.dump(all_players_stats_serializable, f, indent=4, ensure_ascii=False)


# with open('./source/all_players_stats.json', 'r', encoding='utf-8') as f:
#     all_players_stats = json.load(f)

# Convertir el diccionario a DataFrame
players_df = pd.DataFrame.from_dict(all_players_stats, orient='index')
players_df.reset_index(inplace=True)
players_df.rename(columns={'index': 'player_name'}, inplace=True)

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


players_df = players_df[players_df['minutes_total']>=90]
# players_df = players_df[(players_df['npxg_90']>0) & (players_df['xa_90']>0) & (players_df['goals_90']>0)]


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


fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(18, 10))
fig.suptitle(f'Performance of {config.player.name} (EURO 2025)', fontsize=20, color='#15A9D6', y=0.96)
fig.text(x=0.35, y=0.9, s="Sample of wingers of UEFA Women's Euro 2025 (>90min)", fontsize=12,
         style='italic', color='#3f3f3f')
im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
newax_liga = fig.add_axes([0.82, 0.85, 0.13, 0.13], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')
im_player = plt.imread(f'./images/{HIGHLIGHTED_PLAYER.replace(" ", "_").lower()}.png')  # insert local path of the image.
newax_player = fig.add_axes([-0.04, 0.84, 0.15, 0.15], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')

plt.subplots_adjust(left=0.05,
                    bottom=0.1,
                    right=0.95,
                    top=0.8,
                    wspace=0.2,
                    hspace=0.3)

# sns.set_palette("hls", 4)
cmap = 'Blues_r'

for metric in config.metrics:
    utils.create_horizontal_strip_plot_uni(axs[metric.ax[0], metric.ax[1]], players_df, metric.name,
                                           f'{metric.name}_Q', cmap, metric.title, HIGHLIGHTED_PLAYER )
    # utils.create_swarm_plot_with_highlight(axs[metric.ax[0], metric.ax[1]], players_df, metric.name,
    #                                        f'{metric.name}_Q', cmap, metric.title, HIGHLIGHTED_PLAYER)


fig.savefig(f'./output/{HIGHLIGHTED_PLAYER.replace(" ", "_")}.png')
