"""
Plotting events
==============
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch, Sbopen

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

mask_player = (df.type_name == 'Carry') & (df.player_name == player_name)
df_carry = df.loc[mask_player, ['x', 'y', 'end_x', 'end_y']]

# Calculamos la distancia recorrida (Euclidiana)
# Calculation of the distance traveled (Euclidean distance)
df_carry['distance'] = np.sqrt((df_carry.end_x - df_carry.x)**2 + (df_carry.end_y - df_carry.y)**2)

# Filtering those that are greater than 5 meters
# df_carry = df_carry.loc[df_carry['distance'] > 5]

pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)

pitch.arrows(df_carry.x, df_carry.y,
            df_carry.end_x, df_carry.end_y, cmap="Blues", array=df_carry['distance'], ax=ax['pitch'])
# pitch.scatter(df_carry.x, df_carry.y, alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])
fig.suptitle(f"Carries EURO 2025: {player_name}", fontsize = 30)
plt.show()
