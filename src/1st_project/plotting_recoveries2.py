"""
Plotting events
==============
"""

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, Sbopen
from omegaconf import OmegaConf

import utils

config = OmegaConf.load('config.yml')

player_name = config.player.full_name
team_name = config.player.team

parser = Sbopen()
# All matches of the EURO 2025
euro_2025_matches = parser.match(53, 315)

# Recopilar datos de TODAS las jugadoras para la comparación
all_players_recoveries = {}

for index, match in euro_2025_matches.iterrows():
    match_id = match['match_id']
    events, related, freeze, tactics = parser.event(match_id)

    events = events[events['position_name'].isin(config.target_positions)]
    # Obtener jugadoras únicas de este partido
    for player in events['player_name'].dropna().unique():
        if player not in all_players_recoveries:
            all_players_recoveries[player] = {'recoveries': 0, 'minutes': 0}

        player_events = events[events['player_name'] == player]

        # Contar recuperaciones
        recoveries_mask = player_events['type_name'].isin(config.recovery_events)
        recoveries_count = recoveries_mask.sum()

        # Calcular minutos (simplificado - usar eventos del jugador)
        if not player_events.empty:
            player_id = player_events['player_id'].iloc[0]
            minutes = utils.calculate_minutes_played(events, player_id, tactics)
            all_players_recoveries[player]['recoveries'] += recoveries_count
            all_players_recoveries[player]['minutes'] += minutes

# Crear DataFrame para comparación
comparison_df = pd.DataFrame.from_dict(all_players_recoveries, orient='index').reset_index()
comparison_df.rename(columns={'index': 'player_name'}, inplace=True)
comparison_df = comparison_df[comparison_df['minutes'] >= 90].copy()
comparison_df['recoveries_90'] = (comparison_df['recoveries'] / comparison_df['minutes']) * 90

# Calcular z-score en lugar de normalización min-max
metric_mean = comparison_df['recoveries_90'].mean()
metric_std = comparison_df['recoveries_90'].std()

if metric_std > 0:
    comparison_df['recoveries_90_zscore'] = (comparison_df['recoveries_90'] - metric_mean) / metric_std
else:
    comparison_df['recoveries_90_zscore'] = 0  # Valor neutral si no hay variación

comparison_df['recoveries_90_percentile'] = comparison_df['recoveries_90'].rank(pct=True) * 100

# # Normalizar y calcular percentil
# metric_min = comparison_df['recoveries_90'].min()
# metric_max = comparison_df['recoveries_90'].max()
#
# if metric_max - metric_min > 0:
#     comparison_df['recoveries_90_norm'] = (comparison_df['recoveries_90'] - metric_min) / (metric_max - metric_min)
# else:
#     comparison_df['recoveries_90_norm'] = 0.5
# comparison_df['recoveries_90_percentile'] = comparison_df['recoveries_90'].rank(pct=True) * 100

# Eventos del equipo específico
team_euro_2025_matches = euro_2025_matches[
    (euro_2025_matches['home_team_name'] == team_name) |
    (euro_2025_matches['away_team_name'] == team_name)
    ]

# Iterar y acumular eventos de la jugadora destacada
match_events = []
for index, match in team_euro_2025_matches.iterrows():
    match_id = match['match_id']
    events, related, freeze, tactics = parser.event(match_id)
    match_events.append(events.loc[events['player_name'] == player_name])

df = pd.concat(match_events)

mask_player = (df.type_name.isin(config.recovery_events)) & (df.player_name == player_name)
df_recoveries = df.loc[
    mask_player, ['x', 'y', 'type_name', 'sub_type_name', 'outcome_name', 'team_name', 'counterpress', 'minute',
                  'second', 'timestamp']]

mask_ball_recovery = df_recoveries['type_name'] == 'Ball Recovery'
mask_interception = df_recoveries['type_name'] == 'Interception'
mask_duel = (
        (df_recoveries['type_name'] == 'Duel') &
        (df_recoveries['sub_type_name'] == 'Tackle') &
        (df_recoveries['outcome_name'].isin(['Won', 'Success', 'Success in Play', 'Success Out']))
)

df_recoveries = df_recoveries[mask_ball_recovery | mask_interception | mask_duel].copy()
df_recoveries.loc[df_recoveries['type_name'] == 'Duel', 'type_name'] = 'Tackle'
df_recoveries['after_loss'] = False

# Calcular tiempo en segundos
df['total_seconds'] = df['minute'] * 60 + df['second']
df_recoveries['total_seconds'] = df_recoveries['minute'] * 60 + df_recoveries['second']

# Identificar recuperaciones tras pérdida
for idx, recovery in df_recoveries.iterrows():
    recovery_time = recovery['total_seconds']
    recovery_team = recovery['team_name']
    time_window_start = recovery_time - 5

    prev_events = df[
        (df['team_name'] == recovery_team) &
        (df['total_seconds'] >= time_window_start) &
        (df['total_seconds'] < recovery_time)
        ]

    team_loss_events = prev_events[
        (prev_events['type_name'].isin(['Miscontrol', 'Dispossessed', 'Error'])) |
        ((prev_events['type_name'] == 'Duel') & (
            prev_events['outcome_name'].isin(['Lost', 'Lost In Play', 'Lost Out']))) |
        ((prev_events['type_name'] == 'Pass') & (
            prev_events['outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])))
        ]

    if len(team_loss_events) > 0:
        df_recoveries.at[idx, 'after_loss'] = True

df_recoveries.loc[df_recoveries['counterpress'] == True, 'after_loss'] = True
df_recoveries['recovery_type'] = 'Recovery'
df_recoveries.loc[df_recoveries['after_loss'] == True, 'recovery_type'] = 'Recovery After Loss'




# Crear figura con GridSpec
fig = plt.figure(figsize=(12, 10))
# Títulos y textos
fig.suptitle(f"Recoveries of {config.player.name}", fontsize=20, color='#15A9D6', y=0.96)
fig.text(x=0.5, y=0.91, ha='center',
         s=f"Total recoveries: {len(df_recoveries)} ({int(df_recoveries['after_loss'].sum() / len(df_recoveries) * 100)}% after loss)",
         fontsize=12, style='italic', color='#3f3f3f')

# Imágenes
im_liga = plt.imread('./images/euro.png')
newax_liga = fig.add_axes([0.74, 0.88, 0.13, 0.13], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')

im_player = plt.imread(f'./images/{config.player.full_name.replace(" ", "_").lower()}.png')
newax_player = fig.add_axes([0.09, 0.88, 0.14, 0.14], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.1], width_ratios=[1, 1.3,0.15], hspace=-0.2, wspace=0, top=1, left=0.1, right=0.9)

# Subplot principal (pitch)
ax_pitch = fig.add_subplot(gs[0, :])

# Crear pitch
pitch = Pitch(line_color='black')
pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0, title_space=0, endnote_space=0)
pitch.draw(ax=ax_pitch)

# Plot recuperaciones
for i, recovery_type in enumerate(df_recoveries['recovery_type'].unique()):
    mask = df_recoveries['recovery_type'] == recovery_type
    color = '#4292c6' if 'After Loss' not in recovery_type else '#F52780'

    pitch.scatter(df_recoveries.loc[mask, 'x'], df_recoveries.loc[mask, 'y'],
                  label=recovery_type, s=100, color=color, marker='o',
                  alpha=1.0, ax=ax_pitch, edgecolors="black", linewidths=0.5)

ax_pitch.legend(loc='lower left', bbox_to_anchor=(0.035, -0.08), fontsize=11, frameon=False)

# Subplot inferior para comparación
ax_comp = fig.add_subplot(gs[1, 1])

# Usar la función de strip plot
utils.create_horizontal_strip_plot_simple(ax_comp, comparison_df, 'recoveries_90',
                                          'Recoveries/90', player_name)


fig.text(x=0.6, y=0.10, s='* Comparison with other wingers of the EURO 2025.', fontsize=9, color='#08306B', fontstyle='italic')


fig.savefig(f'./output/{config.player.name.replace(" ", "_")}_recoveries.png', bbox_inches='tight', dpi=300)