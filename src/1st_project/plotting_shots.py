"""
Plotting events
==============
"""

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch, Sbopen, VerticalPitch

player_name = 'Caroline Graham Hansen'
team_name = "Norway Women's"


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

mask_player = (df.type_name == 'Shot') & (df.player_name == player_name)
df_shots = df.loc[mask_player, ['x', 'y', 'end_x', 'end_y', 'outcome_name']]


#create pitch
pitch = VerticalPitch(line_color='black', half=True)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

#plotting all shots
# Obtenemos la lista de colores por defecto de matplotlib para asignarlos manualmente
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Definimos una lista de estilos de línea para rotar
linestyles = ['-', '--', '-.', ':']
# Iteramos sobre cada tipo de resultado único en los datos
# Esto permite que matplotlib asigne un color distinto a cada categoría automáticamente
for i, outcome in enumerate(df_shots['outcome_name'].unique()):
    mask = df_shots['outcome_name'] == outcome

    # Seleccionamos un color de la lista (usamos módulo % por si hay más resultados que colores)
    color = colors[i % len(colors)]
    linestyle = linestyles[i % len(linestyles)]

    sc = pitch.scatter(df_shots.loc[mask, 'x'], df_shots.loc[mask, 'y'],
                  label=outcome, s=150, color=color, ax=ax['pitch'], edgecolors="black")


    # Dibujamos la flecha de dirección usando el mismo color
    pitch.lines(df_shots.loc[mask, 'x'], df_shots.loc[mask, 'y'],
                df_shots.loc[mask, 'end_x'], df_shots.loc[mask, 'end_y'],
                color=color, ax=ax['pitch'], lw=5, linestyle=linestyle, comet=True)

# Agregamos una leyenda para saber qué color corresponde a cada resultado
ax['pitch'].legend(loc='lower left', bbox_to_anchor=(0.08, 0.1), fontsize=12)

fig.suptitle(f"Shots EURO 2025: {player_name}", fontsize = 30)
plt.show()


