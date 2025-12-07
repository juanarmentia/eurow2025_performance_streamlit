"""
Plotting events
==============
"""

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch, Sbopen
from omegaconf import OmegaConf

import utils


config = OmegaConf.load('config.yml')

player_name = config.player.full_name
team_name = config.player.team


##############################################################################
# Opening the dataset
# ----------------------------
# We open the data, using SBopen, then we filter the dataframe so that only passes are left,
# This includes remioing throw-ins.

parser = Sbopen()
# All matches of the EURO 2025
euro_2025_matches = parser.match(53,315)
# All matches of a specific team
team_euro_2025_matches = euro_2025_matches[
    (euro_2025_matches['home_team_name'] == team_name) |
    (euro_2025_matches['away_team_name'] == team_name)
]

# Iterate over the dataframe and accumulate events
match_events = []
for index, match in team_euro_2025_matches.iterrows():
    match_id = match['match_id']
    # Get the events for the current match
    events, related, freeze, tactics = parser.event(match_id)
    match_events.append(events.loc[events['player_name'] == player_name])

# Concat all the events in a single dataframe
df = pd.concat(match_events)

# passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')
# carries = df.loc[df['type_name'] == 'Carry']

##############################################################################
# Making the pass map using mplsoccer functions
# ----------------------------
# Again, we filter out passes made by a player.
# Then, we take only the columns needed to plot passes  - coordinates of start and end of a pass.
# We draw a pitch and using arrows method we plot the passes.
# Using scatter method we draw circles where the pass started
# filter the dataset to completed passes for player.

mask_player = (df.type_name.isin(config.recovery_events)) & (df.player_name == player_name)
df_recoveries = df.loc[mask_player, ['x', 'y', 'type_name', 'sub_type_name', 'outcome_name', 'team_name', 'counterpress', 'minute', 'second', 'timestamp']]

mask_ball_recovery = df_recoveries['type_name'] == 'Ball Recovery'
mask_interception = df_recoveries['type_name'] == 'Interception'
mask_duel = (
    (df_recoveries['type_name'] == 'Duel') &
    (df_recoveries['sub_type_name'] == 'Tackle') &
    (df_recoveries['outcome_name'].isin(['Won', 'Success', 'Success in Play', 'Success Out']))
)

df_recoveries = df_recoveries[mask_ball_recovery | mask_interception | mask_duel]

df_recoveries.loc[df_recoveries['type_name'] == 'Duel', 'type_name'] = 'Tackle'

df_recoveries['after_loss'] = False

# Calcular el tiempo total en segundos para cada evento
df['total_seconds'] = df['minute'] * 60 + df['second']
df_recoveries['total_seconds'] = df_recoveries['minute'] * 60 + df_recoveries['second']

# Para cada recuperación, buscar pérdidas del mismo equipo en los 5 segundos previos
for idx, recovery in df_recoveries.iterrows():
    recovery_time = recovery['total_seconds']
    recovery_team = recovery['team_name']

    # Buscar eventos del mismo equipo en los 5 segundos previos
    time_window_start = recovery_time - 5

    prev_events = df[
        (df['team_name'] == recovery_team) &
        (df['total_seconds'] >= time_window_start) &
        (df['total_seconds'] < recovery_time)
        ]

    # Verificar si hubo una pérdida del mismo equipo en esos 5 segundos
    # Tipos de pérdida: Miscontrol, Dispossessed, Duel perdido, Error, Interception del rival
    team_loss_events = prev_events[
        (prev_events['type_name'].isin(['Miscontrol', 'Dispossessed', 'Error'])) |
        ((prev_events['type_name'] == 'Duel') & (
            prev_events['outcome_name'].isin(['Lost', 'Lost In Play', 'Lost Out']))) |
        ((prev_events['type_name'] == 'Pass') & (
            prev_events['outcome_name'].isin(['Incomplete', 'Out', 'Pass Offside'])))
        ]

    if len(team_loss_events) > 0:
        df_recoveries.at[idx, 'after_loss'] = True

# También marcar las que tienen counterpress activado
df_recoveries.loc[df_recoveries['counterpress'] == True, 'after_loss'] = True

df_recoveries['recovery_type'] = 'Recovery'
df_recoveries.loc[df_recoveries['after_loss'] == True, 'recovery_type'] = 'Recovery After Loss'

# Estadísticas
print(f"\nRecuperaciones tras pérdida (5s): {df_recoveries['after_loss'].sum()}")
print(f"Recuperaciones totales: {len(df_recoveries)}")
print(f"Porcentaje tras pérdida: {(df_recoveries['after_loss'].sum() / len(df_recoveries) * 100):.1f}%")


#create pitch
pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.8, title_height=0.06, axis=False,
                     endnote_height=0, title_space=0, endnote_space=0)


# Iteramos sobre cada tipo de resultado único en los datos
# Esto permite que matplotlib asigne un color distinto a cada categoría automáticamente
for i, recovery_type in enumerate(df_recoveries['recovery_type'].unique()):
    mask = df_recoveries['recovery_type'] == recovery_type

    # Usar diferentes marcadores para diferenciar visualmente
    marker = 'o' if 'After Loss' not in recovery_type else 'o'
    alpha = 1.0 if 'After Loss' not in recovery_type else 1.0
    color = '#4292c6' if 'After Loss' not in recovery_type else '#F52780'

    sc = pitch.scatter(df_recoveries.loc[mask, 'x'], df_recoveries.loc[mask, 'y'],
                  label=recovery_type, s=100, color=color, marker=marker,
                  alpha=alpha, ax=ax['pitch'], edgecolors="black", linewidths=0.5)

# Agregamos una leyenda para saber qué color corresponde a cada resultado
ax['pitch'].legend(loc='lower left', bbox_to_anchor=(0.035, -0.05), fontsize=11, frameon=False)

fig.suptitle(f"Recoveries of {config.player.name}", fontsize = 20, color='#15A9D6', y=0.94)
fig.text(x=0.5, y=0.88, ha='center', s=f"Total recoveries: {len(df_recoveries)} ({int(df_recoveries['after_loss'].sum() / len(df_recoveries) * 100)}% after loss)", fontsize=12,
         style='italic', color='#3f3f3f')
im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
newax_liga = fig.add_axes([0.82, 0.85, 0.13, 0.13], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')
im_player = plt.imread(f'./images/{config.player.full_name.replace(" ", "_").lower()}.png')  # insert local path of the image.
newax_player = fig.add_axes([0.015, 0.85, 0.14, 0.14], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')


fig.savefig(f'./output/{config.player.name.replace(" ", "_")}_recoveries.png')


