"""
Plotting events
==============
"""

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Sbopen
from mplsoccer.soccer.pitch import VerticalPitch
from omegaconf import OmegaConf

import utils

config = OmegaConf.load('config.yml')

player_name = config.player.full_name
team_name = "Spain Women's"


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
    match_events.append(events)

# Concat all the events in a single dataframe
df = pd.concat(match_events)

# passes = df.loc[df['type_name'] == 'Pass'].loc[df['sub_type_name'] != 'Throw-in'].set_index('id')
# receptions = df.loc[df['type_name'] == 'Pass'].loc[df['pass_recipient_name'] == player_name]

##############################################################################
# Making the pass map using mplsoccer functions
# ----------------------------
# Again, we filter out passes made by a player.
# Then, we take only the columns needed to plot passes  - coordinates of start and end of a pass.
# We draw a pitch and using arrows method we plot the passes.
# Using scatter method we draw circles where the pass started
# filter the dataset to completed passes for player.

mask_player = (df.type_name == 'Pass') & (df.pass_recipient_name == player_name)
received_passes = df.loc[mask_player, ['x', 'y', 'end_x', 'end_y', 'possession', 'team_name', 'index', 'outcome_name', 'sub_type_name']]

first_2_thirds_progressive = pd.DataFrame()
last_third_progressive = pd.DataFrame()

if not received_passes.empty and 'end_x' in received_passes.columns:
    # Filter successful passes (excluding set pieces)
    successful_received = received_passes[
        (received_passes['outcome_name'].isna()) &
        (~received_passes['sub_type_name'].isin(['Free Kick', 'Corner', 'Kick Off', 'Throw-in']))
        ].copy()

    # List to store progressive passes
    progressive_indices_last_3 = []
    progressive_indices_2_3 = []

    if not successful_received.empty:
        goal_x = 120
        goal_y = 40

        for idx, pass_row in successful_received.iterrows():
            # Calculate distance from start position to goal centre
            distance_before = ((goal_x - pass_row['x']) ** 2 + (goal_y - pass_row['y']) ** 2) ** 0.5

            # Calculate distance from end position to goal centre
            distance_after = ((goal_x - pass_row['end_x']) ** 2 + (goal_y - pass_row['end_y']) ** 2) ** 0.5

            # Calculate reduction in distance
            distance_reduction = distance_before - distance_after

            # Check if the pass reduced distance by at least 25%
            if distance_reduction >= 0.25 * distance_before:
                if pass_row['x'] < 80:
                    progressive_indices_2_3.append(idx)
                else:
                    progressive_indices_last_3.append(idx)


    # df_pass['x_progression'] = df_pass['end_x'] - df_pass['x']
    #
    # first_2_thirds_progressive = df_pass[
    #     (df_pass['x'] < 80) &
    #     (df_pass['x_progression'] >= 10)
    #     ]
    #
    # last_third_progressive = df_pass[
    #     (df_pass['x'] >= 80) &
    #     (df_pass['x_progression'] >= 5)
    #     ]
    #
    # # All receptions in penalty area (regardless of progression)
    # # StatsBomb coordinates: penalty area is approximately x: 102-120, y: 18-62
    # penalty_area_receptions = df_pass[
    #     (df_pass['end_x'] >= 102) &
    #     (df_pass['end_x'] <= 120) &
    #     (df_pass['end_y'] >= 18) &
    #     (df_pass['end_y'] <= 62)
    #     ]

# Select zone based on config
if config.receptions.zone == 'third':
    # Last third progressive + penalty area
    df_receptions = successful_received.loc[progressive_indices_last_3].copy()
elif config.receptions.zone == 'full':
    # All progressive + penalty area
    df_receptions = pd.concat([successful_received.loc[progressive_indices_2_3].copy(), successful_received.loc[progressive_indices_last_3].copy()],
                             axis=0, ignore_index=True)
else:
    df_receptions = pd.DataFrame()

df_receptions['xg_after_reception'] = 0.0

# For each reception
for idx, reception in df_receptions.iterrows():
    possession_id = reception['possession']
    team_name = reception['team_name']
    reception_index = reception['index']

    # Find all shots in the same possession after this reception
    mask_shots_after = (
            (df['type_name'] == 'Shot') &
            (df['possession'] == possession_id) &
            (df['team_name'] == team_name) &
            (df['index'] > reception_index)
    )

    shots_after = df.loc[mask_shots_after]

    # Sum xG from those shots
    if not shots_after.empty and 'shot_statsbomb_xg' in shots_after.columns:
        xg_sum = shots_after['shot_statsbomb_xg'].sum()
        df_receptions.at[idx, 'xg_after_reception'] = xg_sum

# Calculate total xG generated
total_xg = df_receptions['xg_after_reception'].sum()
num_receptions = len(df_receptions)
num_receptions_with_xg = (df_receptions['xg_after_reception'] > 0).sum()

pitch = VerticalPitch(line_color='black', linewidth=1, half=True)
fig, ax = pitch.grid(grid_height=0.9, title_height=0.08, axis=False,
                     endnote_height=0, title_space=0, endnote_space=0)


# Plot arrows colored by xG generated (blue colormap)
if not df_receptions.empty:
    xg_values = df_receptions['xg_after_reception'].values

    # Plot with blue colormap
    arrows = pitch.arrows(
        df_receptions.x,
        df_receptions.y,
        df_receptions.end_x,
        df_receptions.end_y,
        cmap=utils.truncate_colormap('PuRd', minval=0.1, maxval=1.0),  # Blue colormap
        array=xg_values,  # Color based on xG
        width=2,
        ax=ax['pitch']
    )

    # Add colorbar
    cbar = plt.colorbar(arrows, ax=ax['pitch'], orientation='horizontal',
                        pad=0.02, shrink=0.6)
    cbar.set_label('xG generated after reception in the last third', fontsize=10)


# pitch.scatter(df_carry.x, df_carry.y, alpha = 0.2, s = 500, color = "blue", ax=ax['pitch'])
fig.suptitle(f"Progressive Receptions EURO 2025: {config.player.name}", fontsize = 20, color='#15A9D6', y=0.94)
# Add statistics text
stats_text = f"Total xG generated: {total_xg:.2f}\n" \
             f"Progressive receptions: {num_receptions}\n" \
             f"Receptions leading to shots: {num_receptions_with_xg}"

fig.text(0.16, 0.23, stats_text,
         fontsize=11, ha='left', va='bottom',
         style='italic', color='#67001F')

im_liga = plt.imread('./images/euro.png')  # insert local path of the image.
newax_liga = fig.add_axes([0.83, 0.82, 0.15, 0.15], anchor='NE', zorder=1)
newax_liga.imshow(im_liga)
newax_liga.axis('off')
im_player = plt.imread(f'./images/{config.player.full_name.replace(" ", "_").lower()}.png')  # insert local path of the image.
newax_player = fig.add_axes([-0.03, 0.81, 0.18, 0.18], anchor='NE', zorder=1)
newax_player.imshow(im_player)
newax_player.axis('off')
fig.savefig(f'./output/{player_name.replace(" ", "_")}_receptions.png')

