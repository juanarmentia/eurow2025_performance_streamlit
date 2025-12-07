"""
Plotting passes
==============

Making a pass map using Statsbomb data
"""
#importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch, Sbopen

player_name = 'Sara Caroline Seger'

##############################################################################
# Opening the dataset
# ----------------------------
# We open the data, using SBopen, then we filter the dataframe so that only passes are left,
# This includes remioing throw-ins.

parser = Sbopen()
df, related, freeze, tactics = parser.event(69301)

# Player's team
player_team = df.loc[df["player_name"] == player_name, "team_name"].unique()
if len(player_team) == 0:
    raise ValueError(f"No se encontraron eventos para el jugador {player_name}")
player_team = player_team[0]

# Both teams of the match
teams = df["team_name"].unique()

# The opponent team of the player
opponent_team = [t for t in teams if t != player_team][0]

passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')

##############################################################################
# Making the pass map using mplsoccer functions
# ----------------------------
# Again, we filter out passes made by a player.
# Then, we take only the columns needed to plot passes  - coordinates of start and end of a pass.
# We draw a pitch and using arrows method we plot the passes.
# Using scatter method we draw circles where the pass started
# filter the dataset to completed passes for player.

mask_player = (df.type_name == 'Pass') & (df.player_name == player_name)
df_pass = df.loc[mask_player, ['x', 'y', 'end_x', 'end_y']]

pitch = Pitch(line_color='black')
fig, ax = pitch.grid(grid_height=0.9, title_height=0.06, axis=False,
                     endnote_height=0.04, title_space=0, endnote_space=0)
pitch.arrows(df_pass.x, df_pass.y,
            df_pass.end_x, df_pass.end_y, color = "blue", ax=ax['pitch'])
pitch.scatter(df_pass.x, df_pass.y, alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])
fig.suptitle(f"{player_name} passes against {opponent_team}", fontsize = 30)
plt.show()
