import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Sbopen
from scipy.stats import gaussian_kde

from omegaconf import OmegaConf

config = OmegaConf.load('config.yml')

def highlight_player_point(ax, data, x_col, player_name):
    """
    Highlight a specific player's point on a plot by finding their data in the dataset
    and marking it with a distinctive marker. The function is particularly useful when
    visualizing player-specific data and helps in easily identifying the selected player's
    data among others.

    :param ax: Matplotlib axis object where the player's point will be highlighted.
    :param data: pandas DataFrame containing the dataset. It should include a column
        named `player_name` and the x-axis column specified by `x_col`.
    :param x_col: Name of the column in the dataset used for the x-axis values.
    :param player_name: Name of the player as a string to highlight in the plot.
    :return: None
    """
    player_data = data[data['player_name'] == player_name]
    if not player_data.empty:
        ax.scatter(player_data[x_col].values, [0] * len(player_data),
                  s=60, color='#F5CC27', marker='D', zorder=10, edgecolors='white', linewidths=0.5)


def create_swarm_plot_with_highlight(ax, data, x_column, hue_column, palette, title, highlighted_player):
    """
    Create a swarm plot with a highlighted player.

    This function generates a swarm plot using the provided dataset and highlights
    a specific player point for visual emphasis. It customizes the plot by removing
    the y-axis labels and ticks and ensures symmetrical y-axis limits around zero.

    :param ax: matplotlib.axes.Axes; The Axes object on which the swarm plot will be drawn.
    :param data: pandas.DataFrame; The dataset containing the data to be plotted.
    :param x_column: str; The column name in the dataset to be used for the x-axis values.
    :param hue_column: str; The column name in the dataset used to assign color/hue in the plot.
    :param palette: dict or seaborn color palette; A palette used to determine colors in the plot.
    :param title: str; The title to be displayed on the plot.
    :param highlighted_player: str; The identifier or value used to highlight a specific player in the plot.
    :return: None
    """
    sns.swarmplot(x=x_column, data=data, hue=hue_column, palette=palette,
                  legend=False, ax=ax).set(title=title, xlabel=None)
    ax.set(title=title, xlabel=None, ylabel=None)
    ax.set_yticks([])  # Remove y-axis ticks

    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()
    # Calculate the maximum absolute value to create symmetric limits
    y_abs_max = max(abs(y_min), abs(y_max))
    # Set symmetric y-axis limits around 0
    ax.set_ylim(-y_abs_max, y_abs_max)

    highlight_player_point(ax, data, x_column, highlighted_player)


def calculate_minutes_played(events: pd.DataFrame, player_id: int, tactics: pd.DataFrame) -> int:
    """
    Calculate the minutes played by a player in a match.

    Takes into account:
    - Whether the player was a starter or substitute
    - Substitution in (for non-starters)
    - Substitution out (for any player)

    Args:
        events: DataFrame with all match events
        player_id: Player ID
        tactics: DataFrame with starting lineup

    Returns:
        int: Minutes played by the player (0 if didn't play)
    """
    # Calculate max match duration for minutes computation
    max_minute = events['minute'].max()

    # Determine if starter
    is_starter = player_id in tactics['player_id'].values

    min_start = 0 if is_starter else None
    min_end = max_minute

    # Find Substitution event (In)
    if not is_starter:
        sub_in_event = events[(events['type_name'] == 'Substitution') &
                              (events['substitution_replacement_id'] == player_id)]
        if not sub_in_event.empty:
            min_start = sub_in_event['minute'].iloc[0]

    # Find Substitution event (Out)
    sub_out_event = events[(events['type_name'] == 'Substitution') &
                           (events['player_id'] == player_id)]
    if not sub_out_event.empty:
        min_end = sub_out_event['minute'].iloc[0]

    # If min_start is missing (e.g., didn't play or data error), return 0
    if min_start is None:
        minutes_played = 0
    else:
        minutes_played = min_end - min_start

    # Return 0 if negative or zero minutes
    return max(0, minutes_played)


def get_player_data(events: pd.DataFrame, player_id: int, tactics: pd.DataFrame) -> dict:
    """
    Analyzes the performance and contributions of a player during a match based on the given events
    and tactical setup data. The function extracts relevant statistics, such as goals, assists,
    non-penalty expected goals, key passes, progressive actions, and other contributions. The
    results are returned as a dictionary containing various metrics about the player's performance.

    :param events: A DataFrame containing event data of the match.
    :param player_id: An integer representing the unique identifier of the player.
    :param tactics: A DataFrame containing tactical lineup information for the match.

    :return: A dictionary containing key performance metrics of the player. Returns None if the player
             did not play or had zero minutes of valid gameplay.
    """
    # Get player name
    player_name = events[events['player_id'] == player_id]['player_name'].iloc[0]

    player_events = events[events['player_id'] == player_id]

    if player_events.empty:
        return None

    minutes_played = calculate_minutes_played(events, player_id, tactics)

    # Non-penalty goals
    goals = len(player_events[
                    (player_events['type_name'] == 'Shot') &
                    (player_events['outcome_name'] == 'Goal') &
                    (player_events['sub_type_name'] != 'Penalty')
                    ])

    # All shots (excluding penalties)
    shots = len(player_events[
                    (player_events['type_name'] == 'Shot') &
                    (player_events['sub_type_name'] != 'Penalty')
                    ])

    # Non-penalty xG (npxG)
    npxg = player_events[
        (player_events['type_name'] == 'Shot') &
        (player_events['sub_type_name'] != 'Penalty')
        ]['shot_statsbomb_xg'].sum()

    # Assists (Passes leading to a goal)
    assist_column = player_events.get('pass_goal_assist', pd.Series(dtype=bool))
    assists = len(assist_column[assist_column == True])

    # Key Passes (Passes leading to a shot)
    key_pass_column = player_events.get('pass_shot_assist', pd.Series(dtype=bool))
    key_passes = len(key_pass_column[key_pass_column == True])

    # xAssists (expected assists) - sum of xG from shots created by player's passes
    shot_assists = player_events[player_events['pass_assisted_shot_id'].notna()]
    if not shot_assists.empty:
        shot_ids = shot_assists['pass_assisted_shot_id'].values
        xa_value = events[events['id'].isin(shot_ids)]['shot_statsbomb_xg'].sum()
    else:
        xa_value = 0.0

    # Progressive Passes
    # Any successful pass (set pieces excluded) that moves the ball at least 25%
    # of the remaining distance towards the centre of the goal
    pass_events = player_events[player_events['type_name'] == 'Pass'].copy()
    progressive_passes = 0

    if not pass_events.empty and 'end_x' in pass_events.columns:
        # Filter successful passes (excluding set pieces)
        successful_passes = pass_events[
            (pass_events['outcome_name'].isna()) &
            (~pass_events['sub_type_name'].isin(['Free Kick', 'Corner', 'Kick Off', 'Throw-in']))
            ].copy()

        if not successful_passes.empty:
            # StatsBomb uses 120x80 pitch dimensions
            # Goal is at x=120, y=40 (centre)
            goal_x = 120
            goal_y = 40

            for idx, pass_row in successful_passes.iterrows():
                # Calculate distance from start position to goal centre
                distance_before = ((goal_x - pass_row['x']) ** 2 + (goal_y - pass_row['y']) ** 2) ** 0.5

                # Calculate distance from end position to goal centre
                distance_after = ((goal_x - pass_row['end_x']) ** 2 + (goal_y - pass_row['end_y']) ** 2) ** 0.5

                # Calculate reduction in distance
                distance_reduction = distance_before - distance_after

                # Check if the pass reduced distance by at least 25%
                if distance_reduction >= 0.25 * distance_before:
                    progressive_passes += 1

        # successful_passes = pass_events[pass_events['outcome_name'].isna()].copy()
        #
        # if not successful_passes.empty:
        #     successful_passes['x_progression'] = successful_passes['end_x'] - successful_passes['x']
        #
        #     own_half_progressive = successful_passes[
        #         (successful_passes['x'] < 60) &
        #         (successful_passes['x_progression'] >= 10)
        #         ]
        #
        #     opponent_half_progressive = successful_passes[
        #         (successful_passes['x'] >= 60) &
        #         (successful_passes['x_progression'] >= 5)
        #         ]
        #
        #     progressive_passes = len(own_half_progressive) + len(opponent_half_progressive)

    # Progressive Passes Received
    # Count progressive passes where this player was the recipient
    received_passes = events[
        (events['type_name'] == 'Pass') &
        (events['pass_recipient_name'] == player_name) &
        (events['team_name'] == player_events['team_name'].iloc[0] if not player_events.empty else '')
        ].copy()

    progressive_passes_received = 0

    if not received_passes.empty and 'end_x' in received_passes.columns:
        # Filter successful passes (excluding set pieces)
        successful_received = received_passes[
            (received_passes['outcome_name'].isna()) &
            (~received_passes['sub_type_name'].isin(['Free Kick', 'Corner', 'Kick Off', 'Throw-in']))
            ].copy()

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
                    progressive_passes_received += 1

    # Completed Dribbles
    dribbles = len(
        player_events[(player_events['type_name'] == 'Dribble') &
                      (player_events['outcome_name'] == 'Complete')]
    )

    # Carries
    carries = len(player_events[player_events['type_name'] == 'Carry'])

    # Progressive Carries (including dribbles)
    # Any successful carry that moves the ball at least 25%
    # of the remaining distance towards the centre of the goal
    carry_events = player_events[player_events['type_name'] == 'Carry'].copy()
    progressive_carries = 0

    if not carry_events.empty and 'end_x' in carry_events.columns:
        # StatsBomb uses 120x80 pitch dimensions
        # Goal is at x=120, y=40 (centre)
        goal_x = 120
        goal_y = 40

        for idx, carry_row in carry_events.iterrows():
            # Calculate distance from start position to goal centre
            distance_before = ((goal_x - carry_row['x']) ** 2 + (goal_y - carry_row['y']) ** 2) ** 0.5

            # Calculate distance from end position to goal centre
            distance_after = ((goal_x - carry_row['end_x']) ** 2 + (goal_y - carry_row['end_y']) ** 2) ** 0.5

            # Calculate reduction in distance
            distance_reduction = distance_before - distance_after

            # Check if the carry reduced distance by at least 25%
            if distance_reduction >= 0.25 * distance_before:
                progressive_carries += 1


    # DEFENSIVE
    # Interceptions
    interceptions = len(player_events[player_events['type_name'] == 'Interception'])

    ball_recoveries = len(player_events[player_events['type_name'] == 'Ball Recovery'])

    successful_tackles = len(player_events[
                    (player_events['type_name'] == 'Duel') &
                    (player_events['sub_type_name'] == 'Tackle') &
                    (player_events['outcome_name'].isin(['Won', 'Success', 'Success in Play', 'Success Out']))
                  ])

    total_tackles = len(player_events[
                                 (player_events['type_name'] == 'Duel') &
                                 (player_events['sub_type_name'] == 'Tackle')
                                 ])

    return {
        'player_name': player_name,
        'player_id': player_id,
        'minutes': minutes_played,
        'goals': goals,
        'shots': shots,
        'npxg': float(npxg),
        'assists': assists,
        'xa': float(xa_value),
        'key_passes': key_passes,
        'progr_passes': progressive_passes,
        'progr_passes_rec': progressive_passes_received,
        'dribbles': dribbles,
        'carries': carries,
        'progr_carries': progressive_carries,
        'interceptions': interceptions,
        'ball_recoveries': ball_recoveries,
        'successful_tackles': successful_tackles,
        'total_tackles': total_tackles,
    }

def get_match_data(parser: Sbopen, match_id: int, all_players_stats: dict):
    # Get events and tactics
    # tactics contains the starting lineup with positions
    events, related, freeze, tactics = parser.event(match_id)

    # 2. Identify RW/LW players in this match
    # Strategy: Filter events where the player was playing as RM, LM, RW or LW.
    # Note: StatsBomb records the player's position in each event in the 'position_name' column
    # (if available) or we can infer from tactics.

    # events_rw_lw = events[events['position_name'].isin(config.target_positions)]
    # target_player_ids = events_rw_lw['player_id'].dropna().unique()
    target_player_ids = events['player_id'].dropna().unique()

    if len(target_player_ids) == 0:
        return None

    for player_id in target_player_ids:

        player_metrics = get_player_data(events, player_id, tactics)
        player_name = player_metrics.get('player_name')

        if player_name not in all_players_stats:
            all_players_stats[player_name] = {
                'position': [],'matches': 0, 'minutes': 0, 'goals': 0, 'shots': 0,
                'npxg': 0.0, 'assists': 0, 'xa': 0.0, 'key_passes': 0,
                'progr_passes': 0, 'progr_passes_rec': 0, 'dribbles': 0,
                'carries': 0, 'progr_carries': 0, 'interceptions': 0, 'ball_recoveries': 0,
                'successful_tackles': 0, 'total_tackles': 0
            }

        position = None
        if len(tactics[tactics['player_name'] == player_name]) > 0:
            position = tactics[tactics['player_name'] == player_name]['position_name'].iloc[0]
            all_players_stats[player_name]['position'].append(position)

        all_players_stats[player_name]['matches'] += 1
        all_players_stats[player_name]['minutes'] += player_metrics.get('minutes')
        all_players_stats[player_name]['goals'] += player_metrics.get('goals')
        all_players_stats[player_name]['shots'] += player_metrics.get('shots')
        all_players_stats[player_name]['npxg'] += player_metrics.get('npxg')
        all_players_stats[player_name]['assists'] += player_metrics.get('assists')
        all_players_stats[player_name]['xa'] += player_metrics.get('xa')
        all_players_stats[player_name]['key_passes'] += player_metrics.get('key_passes')
        all_players_stats[player_name]['progr_passes'] += player_metrics.get('progr_passes')
        all_players_stats[player_name]['progr_passes_rec'] += player_metrics.get('progr_passes_rec')
        all_players_stats[player_name]['dribbles'] += player_metrics.get('dribbles')
        all_players_stats[player_name]['carries'] += player_metrics.get('carries')
        all_players_stats[player_name]['interceptions'] += player_metrics.get('interceptions')
        all_players_stats[player_name]['ball_recoveries'] += player_metrics.get('ball_recoveries')
        all_players_stats[player_name]['successful_tackles'] += player_metrics.get('successful_tackles')
        all_players_stats[player_name]['total_tackles'] += player_metrics.get('total_tackles')


def truncate_colormap(cmap_name, minval=0.0, maxval=1.0, n=256):
    """
    Truncate a colormap to use only a portion of it.

    Args:
        cmap_name: Name of the colormap
        minval: Minimum value (0.0 to 1.0)
        maxval: Maximum value (0.0 to 1.0)
        n: Number of color steps

    Returns:
        Truncated colormap
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap_name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def create_compact_distribution_plot(ax, df, metric_name, quartile_col, cmap,
                                     title, highlighted_player):
    """
    Versión compacta que combina violinplot + stripplot
    """
    # Violin para mostrar distribución
    parts = ax.violinplot([df[metric_name].dropna()], positions=[0],
                          vert=False, widths=0.7,
                          showmeans=False, showextrema=False, showmedians=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#d0d0d0')
        pc.set_alpha(0.3)

    # Stripplot con jitter reducido
    colors = []
    for idx, row in df.iterrows():
        if row['player_name'] == highlighted_player:
            colors.append('#E8175D')
        else:
            quartile = row[quartile_col]
            if quartile == f'{metric_name}_Q4':
                colors.append('#054A91')
            elif quartile == f'{metric_name}_Q3':
                colors.append('#3E7CB1')
            elif quartile == f'{metric_name}_Q2':
                colors.append('#81A4CD')
            else:
                colors.append('#C1D3E8')

    # Puntos con jitter mínimo
    y_positions = np.random.uniform(-0.15, 0.15, size=len(df))
    ax.scatter(df[metric_name], y_positions, c=colors, s=40,
               alpha=0.7, edgecolors='white', linewidths=0.5, zorder=3)

    # Resaltar jugador destacado
    player_data = df[df['player_name'] == highlighted_player]
    if not player_data.empty:
        ax.scatter(player_data[metric_name].values[0], 0,
                   color='#E8175D', s=200, zorder=5,
                   marker='*', edgecolors='black', linewidths=2)

    # Estadísticas básicas
    median = df[metric_name].median()
    ax.axvline(median, color='black', linestyle='--', linewidth=1.5,
               alpha=0.5, label=f'Mediana: {median:.2f}')

    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def create_horizontal_strip_plot(ax, df, metric_name, title, highlighted_player):
    """
    Versión horizontal ultra-compacta
    """

    all_players_data = df[df['player_name'] != highlighted_player]

    # Get metric values
    metric_values = all_players_data[metric_name].values

    # Calculate density using KDE (Kernel Density Estimation)
    if len(metric_values) > 1:
        kde = gaussian_kde(metric_values)
        density = kde(metric_values)
        # Normalize density between 0 and 1
        density_norm = (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.ones(len(metric_values))

    # Get Blues colormap
    blues_cmap = plt.get_cmap('Blues')

    # Create colors adjusted by density
    # Higher density -> darker blue (higher value in colormap)
    # Lower density -> lighter blue (lower value in colormap)
    colors = []
    for idx in range(len(all_players_data)):
        # Map density to colormap range (0.3 to 0.9 to avoid extremes)
        color_value = 0.3 + (0.6 * density_norm[idx])
        colors.append(blues_cmap(color_value))


    # Very reduced jitter
    y_positions = np.random.uniform(-0.01, 0.01, size=len(all_players_data))

    ax.scatter(all_players_data[metric_name], y_positions, c=colors, s=50,
               alpha=0.7, edgecolors='white', linewidths=0.5)

    # Resaltar jugador
    player_data = df[df['player_name'] == highlighted_player]
    if not player_data.empty:
        ax.scatter(player_data[metric_name].values[0], 0,
                   color='#F5CC27', s=80, zorder=5,
                   marker='D', edgecolors='white', linewidths=1)
        # # Añadir etiqueta
        # ax.text(player_data[metric_name].values[0], 0.25,
        #         config.player.name,
        #         ha='center', fontsize=9, fontweight='bold')

    ax.set_ylim(-0.3, 0.3)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def create_horizontal_strip_plot_uni(ax, order, df, metric_name, title, highlighted_player, z_min_global=None, z_max_global=None):
    """
    Versión horizontal ultra-compacta
    """
    limits = [-0.3, 0.3]
    sep_bottom = 0.03
    sep_top = 0.02
    y_levels = np.linspace(limits[0]+sep_top, limits[1]-sep_bottom, len(config.metrics))

    all_players_data = df[df['player_name'] != highlighted_player]

    # Get metric values
    # metric_values = all_players_data[f'{metric_name}_norm'].values
    metric_values = all_players_data[f'{metric_name}_zscore'].values

    # Calculate density using KDE (Kernel Density Estimation)
    if len(metric_values) > 1:
        kde = gaussian_kde(metric_values)
        density = kde(metric_values)
        # Normalize density between 0 and 1
        density_norm = (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.ones(len(metric_values))

    # Get Blues colormap
    blues_cmap = plt.get_cmap('Blues')

    # Create colors adjusted by density
    # Higher density -> darker blue (higher value in colormap)
    # Lower density -> lighter blue (lower value in colormap)
    colors = []
    for idx in range(len(all_players_data)):
        # Map density to colormap range (0.3 to 0.9 to avoid extremes)
        color_value = 0.3 + (0.6 * density_norm[idx])
        colors.append(blues_cmap(color_value))

    # Very reduced jitter
    y_positions = np.random.uniform(-0.005, 0.005, size=len(all_players_data))
    y_positions = y_positions + y_levels[order]

    # Usar límites globales si se proporcionan
    if z_min_global is not None and z_max_global is not None:
        z_min = z_min_global
        z_max = z_max_global
    else:
        z_min = df[f'{metric_name}_zscore'].min()
        z_max = df[f'{metric_name}_zscore'].max()

    z_range = z_max - z_min
    x_margin = z_range * 0.05

    ax.plot([z_min - x_margin * 0.5, z_max + x_margin * 0.5], [y_levels[order], y_levels[order]],
            color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

    ax.scatter(all_players_data[f'{metric_name}_zscore'], y_positions, c=colors, s=50,
               alpha=0.7, edgecolors='white', linewidths=0.5)

    # ax.plot([-0.01, 1.05], [y_levels[order], y_levels[order]],
    #         color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    #
    # ax.scatter(all_players_data[f'{metric_name}_norm'], y_positions, c=colors, s=50,
    #            alpha=0.7, edgecolors='white', linewidths=0.5)

    # Resaltar jugador
    player_data = df[df['player_name'] == highlighted_player]
    if not player_data.empty:
        ax.scatter(player_data[f'{metric_name}_zscore'].values[0], y_levels[order],
                   color='#F52780', s=80, zorder=5,
                   marker='D', edgecolors='white', linewidths=1)
        # Añadir etiqueta con el valor real (no z-score)
        ax.text(player_data[f'{metric_name}_zscore'].values[0], y_levels[order] - 0.01,
                round(player_data[metric_name].values[0], 2),
                ha='center', fontsize=9, fontweight='bold', color='#08306B')

    # # Añadir línea vertical en la mediana (0.5 para valores normalizados)
    # ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)

    # Añadir línea vertical en la media (z-score = 0)
    ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)

    # Añadir etiqueta del título a la izquierda
    ax.text(z_min - x_margin, y_levels[order], title,
            ha='right', va='center', fontsize=10, fontweight='bold', color='#08306B')

    # Añadir percentil de la jugadora destacada a la derecha
    if not player_data.empty and f'{metric_name}_percentile' in player_data.columns:
        percentile_value = player_data[f'{metric_name}_percentile'].values[0]
        ax.text(z_max + x_margin, y_levels[order], f'P{percentile_value:.0f}',
                ha='left', va='center', fontsize=10, fontweight='bold', color='#F52780')

    ax.set_ylim(limits[1], limits[0])
    ax.set_yticks([])
    ax.set_xlim(z_min - x_margin, z_max + x_margin)
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_xticklabels(['Min', '25%', '50%', '75%', 'Max'], fontsize=9)
    # ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    # Crear etiquetas del eje X basadas en z-scores
    z_ticks = np.linspace(z_min, z_max, 5)
    z_labels = [f'{z:.1f}σ' for z in z_ticks]
    ax.set_xticks(z_ticks)
    ax.set_xticklabels(z_labels, fontsize=9)

    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def create_horizontal_strip_plot_streamlit(ax, order, df, metric_name, n_metrics, title, highlighted_player, z_min_global=None, z_max_global=None):
    """
    Versión horizontal ultra-compacta
    """
    limits = [-0.3, 0.3]
    sep_bottom = 0.03
    sep_top = 0.02
    y_levels = np.linspace(limits[0]+sep_top, limits[1]-sep_bottom, n_metrics)

    all_players_data = df[df['player_name'] != highlighted_player]

    # Get metric values
    # metric_values = all_players_data[f'{metric_name}_norm'].values
    metric_values = all_players_data[f'{metric_name}_zscore'].values

    # Calculate density using KDE (Kernel Density Estimation)
    if len(metric_values) > 1:
        kde = gaussian_kde(metric_values)
        density = kde(metric_values)
        # Normalize density between 0 and 1
        density_norm = (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.ones(len(metric_values))

    # Get Blues colormap
    blues_cmap = plt.get_cmap('Blues')

    # Create colors adjusted by density
    # Higher density -> darker blue (higher value in colormap)
    # Lower density -> lighter blue (lower value in colormap)
    colors = []
    for idx in range(len(all_players_data)):
        # Map density to colormap range (0.3 to 0.9 to avoid extremes)
        color_value = 0.3 + (0.6 * density_norm[idx])
        colors.append(blues_cmap(color_value))

    # Very reduced jitter
    y_positions = np.random.uniform(-0.005, 0.005, size=len(all_players_data))
    y_positions = y_positions + y_levels[order]

    # Usar límites globales si se proporcionan
    if z_min_global is not None and z_max_global is not None:
        z_min = z_min_global
        z_max = z_max_global
    else:
        z_min = df[f'{metric_name}_zscore'].min()
        z_max = df[f'{metric_name}_zscore'].max()

    z_range = z_max - z_min
    x_margin = z_range * 0.05

    ax.plot([z_min - x_margin * 0.5, z_max + x_margin * 0.5], [y_levels[order], y_levels[order]],
            color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)

    ax.scatter(all_players_data[f'{metric_name}_zscore'], y_positions, c=colors, s=50,
               alpha=0.7, edgecolors='white', linewidths=0.5)

    # ax.plot([-0.01, 1.05], [y_levels[order], y_levels[order]],
    #         color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    #
    # ax.scatter(all_players_data[f'{metric_name}_norm'], y_positions, c=colors, s=50,
    #            alpha=0.7, edgecolors='white', linewidths=0.5)

    # Resaltar jugador
    player_data = df[df['player_name'] == highlighted_player]
    if not player_data.empty:
        ax.scatter(player_data[f'{metric_name}_zscore'].values[0], y_levels[order],
                   color='#F52780', s=80, zorder=5,
                   marker='D', edgecolors='white', linewidths=1)
        # Añadir etiqueta con el valor real (no z-score)
        ax.text(player_data[f'{metric_name}_zscore'].values[0], y_levels[order] - 0.025,
                round(player_data[metric_name].values[0], 2),
                ha='center', fontsize=9, fontweight='bold', color='#08306B')

    # # Añadir línea vertical en la mediana (0.5 para valores normalizados)
    # ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)

    # Añadir línea vertical en la media (z-score = 0)
    ax.axvline(x=0.04, color='gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)

    # Añadir etiqueta del título a la izquierda
    ax.text(z_min - x_margin, y_levels[order], title,
            ha='right', va='center', fontsize=8, fontweight='bold', color='#08306B')

    # Añadir percentil de la jugadora destacada a la derecha
    if not player_data.empty and f'{metric_name}_percentile' in player_data.columns:
        percentile_value = player_data[f'{metric_name}_percentile'].values[0]
        ax.text(z_max + x_margin, y_levels[order], f'P{percentile_value:.0f}',
                ha='left', va='center', fontsize=8, fontweight='bold', color='#F52780')

    ax.set_ylim(limits[1], limits[0])
    ax.set_yticks([])
    ax.set_xlim(z_min - x_margin, z_max + x_margin)
    # ax.set_xlim(-0.05, 1.05)
    # ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    # ax.set_xticklabels(['Min', '25%', '50%', '75%', 'Max'], fontsize=9)
    # ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    # Crear etiquetas del eje X basadas en z-scores
    z_ticks = np.linspace(z_min, z_max, 5)
    z_labels = [f'{z:.1f}σ' for z in z_ticks]
    ax.set_xticks(z_ticks)
    ax.set_xticklabels(z_labels, fontsize=9)

    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)



def create_horizontal_strip_plot_simple(ax, df, metric_name, title, highlighted_player):
    """
    Versión simplificada para visualizaciones con 2 métricas
    """
    from scipy.stats import gaussian_kde

    all_players_data = df[df['player_name'] != highlighted_player]

    # Get metric values
    metric_values = all_players_data[f'{metric_name}_zscore'].values

    # Calculate density using KDE (Kernel Density Estimation)
    if len(metric_values) > 1:
        kde = gaussian_kde(metric_values)
        density = kde(metric_values)
        density_norm = (density - density.min()) / (density.max() - density.min())
    else:
        density_norm = np.ones(len(metric_values))

    # Get Blues colormap
    blues_cmap = plt.get_cmap('Blues')

    colors = []
    for idx in range(len(all_players_data)):
        color_value = 0.3 + (0.6 * density_norm[idx])
        colors.append(blues_cmap(color_value))

    # Very reduced jitter
    y_positions = np.random.uniform(-0.005, 0.005, size=len(all_players_data))

    # Usar límites basados en z-scores para consistencia
    z_min = df[f'{metric_name}_zscore'].min()
    z_max = df[f'{metric_name}_zscore'].max()
    z_range = z_max - z_min
    x_margin = z_range * 0.1

    ax.plot([z_min - x_margin * 0.5, z_max + x_margin * 0.5], [0, 0],
            color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=1)

    ax.scatter(all_players_data[f'{metric_name}_zscore'], y_positions, c=colors, s=40,
               alpha=0.7, edgecolors='white', linewidths=0.5)

    # Añadir línea vertical en la mediana (0 para valores z-score)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3, zorder=1, ymin=0.2, ymax=0.8)
    ax.text(0, 0.015, '0σ', ha='center', fontsize=9, color='#08306B')

    # Resaltar jugador
    player_data = df[df['player_name'] == highlighted_player]
    if not player_data.empty:
        ax.scatter(player_data[f'{metric_name}_zscore'].values[0], 0,
                   color='#F52780', s=80, zorder=5,
                   marker='D', edgecolors='white', linewidths=1)
        # ax.text(player_data[f'{metric_name}_norm'].values[0], -0.015,
        #         config.player.name,
        #         ha='center', fontsize=9, color='#F52780')

    # # Añadir etiqueta del título a la izquierda
    # ax.text(-0.05, 0, title,
    #         ha='right', va='center', fontsize=10, fontweight='bold', color='#08306B')

    # Añadir percentil
    if not player_data.empty and f'{metric_name}_percentile' in player_data.columns:
        percentile_value = player_data[f'{metric_name}_percentile'].values[0]
        ax.text(z_max + x_margin, 0, f'P{percentile_value:.0f}',
                ha='left', va='center', fontsize=10, fontweight='bold', color='#F52780')

    ax.set_ylim(-0.02, 0.02)
    ax.set_yticks([])
    ax.set_xlim(z_min - x_margin, z_max + x_margin)
    ax.set_xticks([])
    # ax.set_xticklabels(['Min', '25%', '50%', '75%', 'Max'], fontsize=9)
    # Crear etiquetas del eje X basadas en z-scores
    # z_ticks = np.linspace(0, 0, 1)
    # z_labels = [f'{z:.1f}σ' for z in z_ticks]
    # ax.tick_params(axis='x', length=1)  # Hacer los ticks más cortos
    # ax.set_xticks(z_ticks)
    # ax.set_xticklabels(z_labels, fontsize=9)
    # Ajustar el espacio entre los ticks del eje X


    ax.grid(axis='x', alpha=0.5, linestyle='--', linewidth=0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)