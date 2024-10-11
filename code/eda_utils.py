import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from itertools import product
from typing import List, Dict
import numpy as np


def boxplot_biomass_by_group(orig_df: pd.DataFrame, figure_titles) -> None:
    plt.figure(figsize=(5, 4), dpi=300)
    df = orig_df.copy()
    df['group_num'] = df['group_num'].apply(lambda x: figure_titles[str(x)])
    sns.boxplot(data=df, x='group_num', y='sum_biomass_ug_ml')
    plt.xlabel('Taxonomic Groups', fontsize=12)
    plt.xticks(rotation=90)
    plt.ylabel('Sum Biomass (ug/ml)', fontsize=12)
    plt.tight_layout()
    plt.show()

def violin_biomass_by_group(df: pd.DataFrame) -> None:
    groups = df['group_num'].unique()
    fig, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(10, 7 * len(groups)), sharex=True)
    for i, group in enumerate(groups):
        ax = axes[i]
        group_df = df[df['group_num'] == group]
        sns.violinplot(data=group_df, y='sum_biomass_ug_ml', ax=ax)
        ax.set_title(f"Violin Plot of sum_biomass_ug_ml Group {group}")
        ax.set_ylabel("Sum Biomass (ug/ml)")
        ax.set_yticks(np.linspace(0, group_df['sum_biomass_ug_ml'].max(), 35))
    
    plt.tight_layout()
    plt.show()

def boxplot_by_depth(df: pd.DataFrame, signals: List=None, by_col: str='depth_discrete', figure_titles=None) -> None:
    if not signals:
        signals = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample']

    signal_names_to_nm = {'red': '700 nm',
                            'green': '525 nm',
                            'yellow': '570 nm',
                            'orange': '610 nm',
                            'violet': '370 nm',
                            'brown': '590 nm',
                            'blue': '470 nm',
                            'pressure': 'Pressure',
                            'temp_sample': 'Temperature',
                            'Total conc': 'Total Chl concentration',
                        }
    fig, axes = plt.subplots(nrows=len(signals), ncols=1, figsize=(13, 20), sharex=True)

    for idx, signal in enumerate(signals):
        ax = axes[idx]
        sns.boxplot(x=by_col, y=signal, data=df, ax=ax)
        ax.set_ylabel(signal_names_to_nm[signal])
        ax.set_xlabel(f'{by_col}')
        ax.set_title(f'{signal_names_to_nm[signal]}')

    plt.tight_layout()
    plt.show()

def remove_outliers_IQR(df: pd.DataFrame, q1: float=0.1, q3: float=0.9) -> pd.DataFrame:
    grouped = df.groupby('group_num')
    Q1 = grouped['sum_biomass_ug_ml'].quantile(q1)
    Q3 = grouped['sum_biomass_ug_ml'].quantile(q3)
    IQR = Q3 - Q1

    # Define a function to filter outliers based on the IQR
    def filter_outliers(group):
        group_num = group.name
        Q1_val = Q1[group_num]
        Q3_val = Q3[group_num]
        iqr_val = IQR[group_num]
        return group[(group['sum_biomass_ug_ml'] >= Q1_val - 1.5 * iqr_val) &
                    (group['sum_biomass_ug_ml'] <= Q3_val + 1.5 * iqr_val)]

    filtered_df = grouped.apply(filter_outliers).reset_index(drop=True)

    return filtered_df

def correlation_per_group(df: pd.DataFrame) -> None:
    columns_to_drop = ['year', 'month', 'Depth']
    correlations_per_group = df.drop(columns=columns_to_drop).groupby('group_num').corr()

    for group_number in df.group_num.unique():
        correlation_for_group = correlations_per_group.loc[group_number]

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_for_group, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f"Correlation Heatmap for Group {group_number}")
        plt.show()


def plot_fluorprobe_prediction(df: pd.DataFrame, fluor_groups_map: Dict) -> None:
    unique_depths = df['Depth'].unique()
    depth_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_depths)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for i, group_num in enumerate(fluor_groups_map.keys()):
        row = i // 2
        col = i % 2
        
        group_y_test = df[df['group_num'] == group_num]['sum_biomass_ug_ml']
        group_y_fluor_pred = df[df['group_num'] == group_num][fluor_groups_map[group_num]]
        group_depth = df[df['group_num'] == group_num]['Depth']

        depth_color_idx = [np.where(unique_depths == d)[0][0] for d in group_depth]

        axes[row, col].scatter(group_y_test, group_y_fluor_pred, c=depth_colors[depth_color_idx], alpha=0.5)
        axes[row, col].plot([group_y_test.min(), group_y_test.max()], [group_y_test.min(), group_y_test.max()], 'r--', lw=2)  
        axes[row, col].set_xlabel('Actual Test Values')
        axes[row, col].set_ylabel('Fluor Predicted Values')
        axes[row, col].set_title(f'Group {group_num} - Actual vs. Fluor Predicted')

    legend_handles = []
    for depth, color in zip(unique_depths, depth_colors):
        legend_handles.append(plt.scatter([], [], c=color, label=depth))

    fig.legend(handles=legend_handles, title='Depth', loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()

def plot_corr_per_feature_per_group(df: pd.DataFrame, fluor_groups_map: Dict) -> None:
    signal_cols = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample']
    for group_num in fluor_groups_map.keys():
        fig, axes = plt.subplots(5, 2, figsize=(16, 12))

        for i, col_name in enumerate(signal_cols):
            row = i % 5
            col = i % 2

            group_y = df[df['group_num'] == group_num]['sum_biomass_ug_ml']
            feat = df[df['group_num'] == group_num][col_name]

            axes[row, col].scatter(group_y, feat)
            axes[row, col].set_xlabel('Biomass [ug/ml]')
            axes[row, col].set_ylabel(f'Signal {col_name}')

        fig.suptitle(f'Group {group_num}')
        plt.tight_layout()
        plt.show()



def integrate_values_over_depth(df, value_columns, integrate_above_thermocline=False):
   
    df = df.dropna()

    all_depths = np.arange(int(df["Depth"].min()), int(df["Depth"].max()) + 1)

    all_combinations = pd.DataFrame([(week, year, month, group_num_title, depth) for week, year, month, group_num_title in df[['week', 'year', 'month', 'group_num_title']].drop_duplicates().values
                                     for depth in all_depths], columns=['week', 'year', 'month', 'group_num_title', 'Depth'])

    merged_df = pd.merge(all_combinations, df, how='left', on=['week', 'year', 'month', 'group_num_title', 'Depth'])

    merged_df.sort_values(['week', 'year', 'month', 'group_num_title', 'Depth'], inplace=True)

    interpolated_df = merged_df.interpolate(method='linear', axis=0)

    if not integrate_above_thermocline:
        integrated_df = interpolated_df.groupby(['week', 'year', 'month', 'group_num_title'])[value_columns].sum().reset_index()
    else:
        thermocline_depth = integrate_above_thermocline
        integrated_df = interpolated_df[interpolated_df['Depth'] <= thermocline_depth].groupby(['week', 'year', 'month', 'group_num_title'])[value_columns].sum().reset_index()

    return integrated_df



def groups_pie_chart(orig_df: pd.DataFrame, figure_titles: Dict, custom_palette) -> None:
    df = orig_df.copy()
    
    df['group_num_title'] = df['group_num'].apply(lambda x: figure_titles[str(x)])

    df = integrate_values_over_depth(df, value_columns=["sum_biomass_ug_ml"], integrate_above_thermocline=15)
    random_seed = 42
    group_counts = df.groupby('group_num_title')['sum_biomass_ug_ml'].sum()

    plt.figure(figsize=(5, 4), dpi=300)
    plt.rcParams['font.size'] = 9
    plt.pie(group_counts, labels=sorted(group_counts.index), autopct='%1.1f%%', startangle=140, colors=[custom_palette[group] for group in sorted(group_counts.index)])
    plt.axis('equal')  
    plt.legend(bbox_to_anchor=(1.3, -0.1), loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.show()

def groups_biomass_by_column(orig_df: pd.DataFrame, figure_titles: Dict, custom_palette, col_to_groupby) -> None:
    df = orig_df.copy()
    df['group_num_title'] = df['group_num'].apply(lambda x: figure_titles[str(x)])
    
    df = df[df["Depth"] <= 20]
    
    grouped_biomass_df = df.groupby(['group_num', col_to_groupby])['sum_biomass_ug_ml'].sum().reset_index()
    grouped_biomass_df['total_sum'] = grouped_biomass_df.groupby(col_to_groupby)['sum_biomass_ug_ml'].transform('sum')
    grouped_biomass_df['percentage'] = (grouped_biomass_df['sum_biomass_ug_ml'] / grouped_biomass_df['total_sum'])
    grouped_biomass_df['group_num_title'] = grouped_biomass_df['group_num'].apply(lambda x: figure_titles[str(x)])

    df_pivot = grouped_biomass_df.pivot(index=col_to_groupby, columns='group_num_title', values='percentage')

    if col_to_groupby == 'month':
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        plt.rcParams['font.size'] = 12
    else:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        plt.rcParams['font.size'] = 12
    df_pivot.plot(kind='bar', stacked=True, color=[custom_palette[group] for group in sorted(grouped_biomass_df['group_num_title'].unique())], ax=ax)

    ax.set_xlabel('')
    if col_to_groupby == 'Depth':
        ax.set_xticklabels(['0-3', '3-5', '5-10', '10-15', '15-20'])
    ax.set_ylabel('Fraction of Total Biomass (ug/ml)')
    ax.get_legend().remove()
    ax.tick_params(axis='x', rotation=0)
    plt.show()
