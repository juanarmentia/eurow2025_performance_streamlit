import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mplsoccer import Sbopen
from omegaconf import OmegaConf

import utils

# Carga configuración (jugador destacado, etc.)
config = OmegaConf.load('config.yml')
HIGHLIGHTED_PLAYER = config.player.full_name

# 1) Recolectar partidos (UEFA Women's Euro 2025: comp=53, season=315)
parser = Sbopen()
matches = parser.match(53, 315)
match_ids = matches['match_id'].unique()

# 2) Construir métricas agregadas por jugadora
all_players_stats = {}
for match_id in match_ids:
    utils.get_match_data(parser, match_id, all_players_stats)  # ya calcula minutes y progr_passes

players_df = pd.DataFrame.from_dict(all_players_stats, orient='index').reset_index()
players_df.rename(columns={'index': 'player_name'}, inplace=True)

# 3) Calcular métricas por 90
players_df['minutes_total'] = players_df['minutes']
players_df['minutes_90'] = players_df['minutes'] / 90
players_df = players_df[players_df['minutes_total'] >= 90].copy()
players_df['progr_passes_90'] = players_df['progr_passes'] / players_df['minutes_90']

# 4) Calcular % de pases completados por jugadora desde los eventos
#    (éxito en StatsBomb: outcome_name es NaN; excluimos saques de banda y jugadas a balón parado comunes)
pass_attempts = {}
pass_completed = {}

for match_id in match_ids:
    events, related, freeze, tactics = parser.event(match_id)

    passes = events[events['type_name'] == 'Pass'].copy()
    passes = passes[~passes['sub_type_name'].isin(['Throw-in', 'Free Kick', 'Corner', 'Kick Off'])]

    # Intentos por jugadora (quien ejecuta el pase)
    attempts_by_player = passes.groupby('player_name').size()
    # Completados por jugadora (outcome_name NaN => completado)
    completed_by_player = passes[passes['outcome_name'].isna()].groupby('player_name').size()

    for p, n in attempts_by_player.items():
        pass_attempts[p] = pass_attempts.get(p, 0) + int(n)
    for p, n in completed_by_player.items():
        pass_completed[p] = pass_completed.get(p, 0) + int(n)

passes_df = pd.DataFrame({
    'player_name': list(set(pass_attempts.keys()) | set(pass_completed.keys())),
})
passes_df['pass_attempts'] = passes_df['player_name'].map(lambda p: pass_attempts.get(p, 0))
passes_df['pass_completed'] = passes_df['player_name'].map(lambda p: pass_completed.get(p, 0))
passes_df['pass_completion_pct'] = passes_df.apply(
    lambda r: 100 * r['pass_completed'] / r['pass_attempts'] if r['pass_attempts'] > 0 else 0,
    axis=1
)

# 5) Unir con players_df (que contiene minutes y progr_passes)
players_df = players_df.merge(passes_df[['player_name', 'pass_completion_pct', 'pass_attempts']],
                              on='player_name', how='left')

# 6) Scatter plot: X = % completados, Y = progresivos/90
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(12, 10))
fig.suptitle(f'Pass % vs Progressive passes/90 of {config.player.name}', fontsize=20, color='#15A9D6', y=0.94)
fig.text(x=0.5, y=0.88, ha='center', s="Sample of wingers of UEFA Women's Euro 2025 (>90min)", fontsize=12,
         style='italic', color='#3f3f3f')

plt.subplots_adjust(left=0.08,
                    bottom=0.07,
                    right=0.93,
                    top=0.82,
                    wspace=0.2,
                    hspace=0.3)

sns.set_theme(style="whitegrid", palette="pastel")
sns.scatterplot(
    data=players_df,
    x='pass_completion_pct',
    y='progr_passes_90',
    size='pass_attempts',  # opcional: tamaño por volumen de pases
    sizes=(40, 250),
    alpha=1.0,
    edgecolor='black',
    linewidth=0.5,
    color='#4292c6',
    legend='brief'
)

# Añadir grid
axs.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Calcular y añadir líneas de mediana
median_x = players_df['pass_completion_pct'].median()
median_y = players_df['progr_passes_90'].median()

axs.axvline(median_x, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
axs.axhline(median_y, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

# Añadir etiqueta a la línea de mediana del eje Y
axs.text(axs.get_xlim()[0] + 1, median_y + 0.15, f'Median: {median_y:.2f}',
         fontsize=9, color='gray', va='bottom', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))
axs.text(median_x, axs.get_ylim()[0] + 0.15, f'Median: {median_x:.1f}%',
         fontsize=9, color='gray', va='bottom', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.7))


# Resaltar jugadora destacada (si existe en el DF)
if (players_df['player_name'] == HIGHLIGHTED_PLAYER).any():
    row = players_df.loc[players_df['player_name'] == HIGHLIGHTED_PLAYER].iloc[0]
    axs.scatter(row['pass_completion_pct'], row['progr_passes_90'],
               s=250, color='#F52780', edgecolor='black', linewidth=1.0, zorder=5)
    axs.annotate(f'{config.player.name}\n({int(row['pass_completion_pct'])}% -- {round(row['progr_passes_90'],2)} ProgP90.)',
                (row['pass_completion_pct'], row['progr_passes_90']),
                textcoords='offset points', xytext=(8, -30), ha='left', fontsize=10, color='#F52780')

    # Configurar la leyenda con título y más espacio
handles, labels = axs.get_legend_handles_labels()
axs.legend(handles, labels, title='Pass Attempts', frameon=False,
          loc='upper left', labelspacing=1.2, title_fontsize=11, fontsize=10)

axs.set_xlabel('Pass completion (%)', size=11)
axs.set_ylabel('Progressive passes per 90', size=11)
# ax.set_title(f'Pass completion vs Progressive passes/90 — Euro 2025 (≥90 min)')
# plt.tight_layout()

im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
newax_liga = fig.add_axes([0.8, 0.85, 0.13, 0.13], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')
im_player = plt.imread(f'./images/{HIGHLIGHTED_PLAYER.replace(" ", "_").lower()}.png')  # insert local path of the image.
newax_player = fig.add_axes([0.04, 0.84, 0.15, 0.15], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')

fig.savefig(f'./output/{HIGHLIGHTED_PLAYER.replace(" ", "_")}_passes_perc_prog.png')