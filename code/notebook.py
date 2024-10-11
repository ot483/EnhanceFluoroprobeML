#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from data_utils import *
from eda_utils import *
from train_utils import *
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import pickle
import os

import warnings
warnings.filterwarnings("ignore")

figure_titles = {
    'red': '700 nm',
    'green': '525 nm',
    'yellow': '570 nm',
    'orange': '610 nm',
    'violet': '370 nm',
    'brown': '590 nm',
    'blue': '470 nm',
    'pressure': 'Depth (m)',
    'temp_sample': 'Temperature',
    'Total conc': 'Total Chl concentration',
    '2' : 'bluegreens (Cyanobacteria)',
    '3' : 'diatoms (Bacillariophyceae)',
    '4' : 'green algea (Chlorophyta + Charophyta)',
    '5' : 'dinoflagellates (Dinophyceae)',
    '6' : 'cryptophytes (Cryptista)',
    'month': 'Month'
}

BaseFolder = "/where/the/scripts/and/files/location/"
phyt_cod_path = BaseFolder+'phytoplankton_codes_table.csv'
phyto_path = BaseFolder + 'phytoplakton_table.csv'
fp_path = BaseFolder + 'FP_dataset.csv'

biomass_by_week_year_group = get_biomass_data(phyt_cod_path, phyto_path)

fp_df = get_fluorprobe_data(fp_path)

biomass_test = biomass_by_week_year_group[
    (biomass_by_week_year_group['year'].isin([2017,2018,2019,2020,2021,2022])) & (biomass_by_week_year_group['week'].isin( list(range(1,52,4)) ))
                                          ]
biomass_train = biomass_by_week_year_group[~biomass_by_week_year_group.index.isin(biomass_test.index)]

fp_test = fp_df[
    (fp_df['year'].isin([2017,2018,2019,2020,2021,2022])) & (fp_df['week'].isin( list(range(1,52,4)) ))
                                          ]
fp_train = fp_df[~fp_df.index.isin(fp_test.index)]

merged_train = merge_fp_biomass_df(fp_train, biomass_train, is_train=True) # Merging fully
merged_test = merge_fp_biomass_df(fp_test, biomass_test, is_train=False) # Merging only closest records by depth

print(len(merged_train))
print(len(merged_test))

merged_train = merged_train.drop(['Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta'], axis=1).reset_index(drop=True)

fluor_groups_map = {
    2: 'Bluegreen',
    3: 'Diatoms',
    4: 'Green Algae',
    6: 'Cryptophyta'
}

fluor_test_df = merged_test[['group_num', 'month', 'week', 'year', 'Depth', 'sum_biomass_ug_ml', 'Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta']].reset_index(drop=True)
merged_test = merged_test.drop(['Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta'], axis=1).reset_index(drop=True)

### Saving original train df for later tests
orig_merged_train = merged_train.copy()

orig_merged_train.loc[orig_merged_train['Depth'] >= 21, 'Depth'] = 30

fluor_train_df = merge_fp_biomass_df(fp_train.drop('Depth', axis=1), biomass_train, is_train=False)
fluor_train_df = fluor_train_df[['group_num', 'month', 'week', 'year', 'Depth', 'sum_biomass_ug_ml', 'Green Algae', 'Bluegreen', 'Diatoms', 'Cryptophyta']].reset_index(drop=True)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

color_palette = sns.color_palette("coolwarm", as_cmap=True)

###############FIGURE 2###################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def corr_fp_biomass_by_depth(df):
    relevant_columns = ['Depth', 'group_num', 'sum_biomass_ug_ml'] + [fluor_groups_map[group_num] for group_num in fluor_groups_map.keys()]
    df_relevant = df[relevant_columns]
    df_relevant.rename(columns={x: figure_titles[str(x)] for x in fluor_groups_map.keys()}, inplace=True)

    correlation_results = {}

    for group_num, column_name in fluor_groups_map.items():
        group_df = df_relevant[df_relevant['group_num'] == group_num]

        group_correlations = {}

        grouped_by_depth = group_df.groupby('Depth')

        for depth, group_data in grouped_by_depth:
            correlation = group_data[['sum_biomass_ug_ml', column_name]].corr().iloc[0, 1]
            group_correlations[depth] = correlation

        correlation_results[column_name] = group_correlations
    
    correlation_results["cryptophytes (Cryptista)"] = correlation_results.pop('Cryptophyta')
    correlation_results["bluegreens (Cyanobacteria)"] = correlation_results.pop('Bluegreen')
    correlation_results["diatoms (Bacillariophyceae)"] = correlation_results.pop('Diatoms')
    correlation_results["green algea (Chlorophyta + Charophyta)"] = correlation_results.pop('Green Algae')

#     0-3, 3-5, 5-10, 10-15, 15-20, and 20-43
    for group_name in correlation_results.keys():
        correlation_results[group_name]['0-3'] = correlation_results[group_name].pop(3)
        correlation_results[group_name]['3-5'] = correlation_results[group_name].pop(5)
        correlation_results[group_name]['5-10'] = correlation_results[group_name].pop(10)
        correlation_results[group_name]['10-15'] = correlation_results[group_name].pop(15)
        correlation_results[group_name]['15-20'] = correlation_results[group_name].pop(20)
        correlation_results[group_name]['20-43'] = correlation_results[group_name].pop(30)
    
    plt.figure(figsize=(10, 8), dpi=300)  
    
    heatmap_data = pd.DataFrame(correlation_results)
    
    sorted_columns = sorted(heatmap_data.columns, key=lambda x: x)

    heatmap_data = heatmap_data[sorted_columns]
    heatmap_data = heatmap_data.T[['0-3', '3-5', '5-10', '10-15', '15-20', '20-43']].T
    
    ax = sns.heatmap(heatmap_data, annot=True, cmap=color_palette, fmt=".2f", vmin=-1, annot_kws={"size": 12})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.ylabel('Depth', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()   
    return correlation_results

correlation_results_depth_FP = corr_fp_biomass_by_depth(fluor_train_df)


############By month

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def corr_fp_biomass_by_month(df):
    relevant_columns = ['month', 'group_num', 'sum_biomass_ug_ml'] + [fluor_groups_map[group_num] for group_num in fluor_groups_map.keys()]
    df_relevant = df[relevant_columns]

    correlation_results = {}

    for group_num, column_name in fluor_groups_map.items():
        group_df = df_relevant[df_relevant['group_num'] == group_num]

        group_correlations = {}

        grouped_by_month = group_df.groupby('month')

        for month, group_data in grouped_by_month:
            correlation = group_data[['sum_biomass_ug_ml', column_name]].corr().iloc[0, 1]
            group_correlations[month] = correlation

        correlation_results[column_name] = group_correlations

    correlation_results["cryptophytes (Cryptista)"] = correlation_results.pop('Cryptophyta')
    correlation_results["bluegreens (Cyanobacteria)"] = correlation_results.pop('Bluegreen')
    correlation_results["diatoms (Bacillariophyceae)"] = correlation_results.pop('Diatoms')
    correlation_results["green algea (Chlorophyta + Charophyta)"] = correlation_results.pop('Green Algae')

    plt.figure(figsize=(10, 8), dpi=300)  
    heatmap_data = pd.DataFrame(correlation_results)
    sorted_columns = sorted(heatmap_data.columns, key=lambda x: x)

    heatmap_data = heatmap_data[sorted_columns]
    
    ax = sns.heatmap(heatmap_data, annot=True, cmap=color_palette, fmt=".2f", vmin=-1, annot_kws={"size": 12})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.ylabel('Month', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return correlation_results

correlation_results_month_FP = corr_fp_biomass_by_month(fluor_train_df)

#########Fig BOXPLOTS#################################

merged_train = merged_train[merged_train['sum_biomass_ug_ml'] >= 0]

merged_train.loc[merged_train['Depth'] >= 21, 'Depth'] = 21
merged_test.loc[merged_test['Depth'] >= 21, 'Depth'] = 21

signals = ['red', 'green', 'yellow', 'orange', 'violet', 'brown', 'blue', 'pressure', 'temp_sample', 'Total conc']


boxplot_by_depth(merged_train, signals, by_col='Depth')


###################################

signals.remove('pressure')
signals.remove('temp_sample')
signals.remove('Total conc')
boundaries = {
    'red': {'lower_bound': 0, 'upper_bound': 120},
    'green': {'lower_bound': 0, 'upper_bound': 100},
    'yellow': {'lower_bound': 0, 'upper_bound': 40},
    'orange': {'lower_bound': 0, 'upper_bound': 60},
    'violet': {'lower_bound': 0, 'upper_bound': 150},
    'brown': {'lower_bound': 0, 'upper_bound': 80},
    'blue': {'lower_bound': 0, 'upper_bound': 220},
}

filter_signals_by_boundaries(merged_train, signals, boundaries)

boxplot_by_depth(merged_train, signals, by_col='Depth')


######Figure 1############

custom_palette = {'bluegreens (Cyanobacteria)': 'blue', 'diatoms (Bacillariophyceae)': 'orange', 'green algea (Chlorophyta + Charophyta)': 'green',
                   'dinoflagellates (Dinophyceae)': 'red', 'cryptophytes (Cryptista)': 'cyan'}

groups_pie_chart(biomass_by_week_year_group, figure_titles, custom_palette)
groups_biomass_by_column(biomass_by_week_year_group, figure_titles, custom_palette, 'Depth')
groups_biomass_by_column(biomass_by_week_year_group, figure_titles, custom_palette, 'month')

merged_train = merged_train.drop(['year', 'Depth', 'depth', 'week'], axis=1)
orig_merged_train = orig_merged_train.drop(['year', 'Depth', 'depth', 'week'], axis=1)

################## Hyperparameter Grid Search
pickled_params = BaseFolder+'best_grid_params_latest_step_2_clean.pkl'
do_search = not os.path.exists(pickled_params)

def two_scorer(mse=False):
    score = mean_squared_error if mse else r2_score
    return make_scorer(score, greater_is_better=mse==False)


if do_search:
    param_grid_per_model = {
        'xgb': {
        "model__n_estimators": [100, 300, 500], 
        "model__max_depth": [2, 3, 5], 
        "model__alpha": [0.01, 0.1, 1, 10], 
        "model__lambda": [0, 0.2, 0.8], 
    },
    'svr':  {
        "model__C": [0.001, 0.1, 1, 10, 100],
        "model__tol": [1e-4, 1e-3, 1e-2] 
    },
    'rf': {
        "model__n_estimators": [100, 300, 500], 
        "model__max_depth": [2, 3, 5], 
        "model__ccp_alpha": [0.01, 0.1, 1, 10],
    }}
    
    best_group_params_per_model = {}
    for model_name, param_grid in param_grid_per_model.items():
        best_group_params_per_model[model_name] = grid_search_cv(model_name, merged_train, param_grid=param_grid,
                                                                scoring_method=two_scorer(mse=True))
    
    with open(pickled_params, 'wb') as f:
        pickle.dump(best_group_params_per_model, f)
    
    best_group_params_per_model_fixed = {}
    for model_name, kwargs_per_group in best_group_params_per_model.items():
        best_group_params_per_model_fixed[model_name] = {}
        for group_num in kwargs_per_group.keys():
            best_group_params_per_model_fixed[model_name][group_num] = {}
            kwargs = kwargs_per_group[group_num]
            for param, val in kwargs.items():
                fixed_param = param.split('model__')[1]
                best_group_params_per_model_fixed[model_name][group_num][fixed_param] = val


    with open(pickled_params, 'wb') as f:
        pickle.dump(best_group_params_per_model_fixed, f)

else:
    with open(pickled_params, 'rb') as f:
        best_group_params_per_model_fixed = pickle.load(f)


############################################

regression_models = {}
for model_name, best_group_params_fixed in best_group_params_per_model_fixed.items():
    regression_models[model_name], _ = train(model_name, merged_train, group_kwargs=best_group_params_per_model_fixed, test_size=0.0,
                                             biomass_fn=np.log)
    
comparing_df = compare_all_models(regression_models,
                                  merged_test.drop(['year', 'Depth', 'week'], axis=1),
                                  fluor_test_df,
                                  fluor_groups_map,
                                  predictions_fn=np.exp,
                                  model_names=['xgb', 'svr', 'rf']
                                  )

cleaned_res = compare_by_mpe(merged_test, regression_models, predict_cols=merged_train.drop(['sum_biomass_ug_ml', 'group_num'], axis=1).columns, predict_fn=np.exp)

fp_res = calc_mpe_fp(fluor_test_df, with_group_5=False)

all_results = pd.concat([fp_res, cleaned_res]).reset_index(drop=True)
all_results

temp_res = all_results.rename(columns={k: figure_titles[str(k)] for k in all_results.columns if k != 'Model'})

temp_res['Mean'] = temp_res.mean(axis=1)
temp_res['std'] = temp_res.std(axis=1)
temp_res

# =============================================================================
# 
# def bar_plot_all_models_cleaned_no_smogn(df):
#     selected_columns = [2, 3, 4, 5, 6]
#     selected_models = ['xgb', 'svr', 'rf', 'FP']
#     selected_sources = ['FP', 'Cleaned']
#     filtered_df = df[df['Model'].isin(selected_models)]
# 
#     # Set up the figure and axis for plotting
#     fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
#     plt.xticks(rotation=90)
#     
#     # Width of each bar
#     bar_width = 0.15
#     index = np.arange(len(selected_columns))
# 
#     # Iterate through each model and create a grouped bar plot
#     for i, model in enumerate(selected_models):
#         model_data = filtered_df[filtered_df['Model'] == model]
#         y = model_data[selected_columns].values[0]  # Corresponding values for the model
#         ax.bar(index + i * bar_width, y, bar_width, label=model)
# 
#     # Set labels and title
#     ax.set_ylabel('Mean Proportion Error (%)', fontsize=12)
#     ax.set_xticks(index + bar_width * (len(selected_models) - 1) / 2)
#     ax.set_xticklabels(figure_titles[str(x)] for x in selected_columns)
#     
#     # Show the plot
#     plt.yticks(fontsize=12)
#     plt.xticks(fontsize=12, rotation=45)
#     plt.show()
# 
# 
# bar_plot_all_models_cleaned_no_smogn(all_results)
# =============================================================================


# =============================================================================
# ### Calc RMSPE
# 
# # Calculate RMSPE for cleaned data
# cleaned_res_rmspe = compare_by_rmspe(merged_test, regression_models, predict_cols=merged_train.drop(['sum_biomass_ug_ml', 'group_num'], axis=1).columns, predict_fn=np.exp)
# 
# # Calculate RMSPE for FP data
# fp_res_rmspe = calc_rmspe_fp(fluor_test_df, with_group_5=False)
# 
# # Combine results
# all_results_rmspe = pd.concat([fp_res_rmspe, cleaned_res_rmspe]).reset_index(drop=True)
# 
# # Rename columns
# temp_res_rmspe = all_results_rmspe.rename(columns={k: figure_titles[str(k)] for k in all_results_rmspe.columns if k != 'Model'})
# 
# # Calculate mean and standard deviation
# temp_res_rmspe['Mean'] = temp_res_rmspe.mean(axis=1)
# temp_res_rmspe['std'] = temp_res_rmspe.std(axis=1)
# 
# 
# def bar_plot_all_models_cleaned_no_smogn_rmspe(df):
#     # Filter the DataFrame to include only the desired columns and models
#     selected_columns = [2, 3, 4, 5, 6]
#     selected_models = ['xgb', 'svr', 'rf', 'FP']
#     filtered_df = df[df['Model'].isin(selected_models)]
# 
#     # Set up the figure and axis for plotting
#     fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
#     plt.xticks(rotation=90)
#     
#     # Width of each bar
#     bar_width = 0.15
#     index = np.arange(len(selected_columns))
# 
#     # Iterate through each model and create a grouped bar plot
#     colors = ['orange', 'blue', 'green', 'red']
#     for i, (model, color) in enumerate(zip(selected_models, colors)):
#         model_data = filtered_df[filtered_df['Model'] == model]
#         if not model_data.empty:
#             y = model_data[selected_columns].values[0]  # Corresponding values for the model
#             ax.bar(index + i * bar_width, y, bar_width, label=model, color=color)
# 
#     # Set labels and title
#     ax.set_ylabel('Root Mean Square Percentage Error (%)', fontsize=12)
#     ax.set_xticks(index + bar_width * (len(selected_models) - 1) / 2)
#     ax.set_xticklabels([figure_titles.get(str(x), str(x)) for x in selected_columns])
#     
#     # Show the plot
#     plt.yticks(fontsize=12)
#     plt.xticks(fontsize=12, rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# 
# # Create the bar plot
# bar_plot_all_models_cleaned_no_smogn_rmspe(all_results_rmspe)
# =============================================================================

### Calc MSE

cleaned_res_mse = compare_by_mse(merged_test, regression_models, predict_cols=merged_train.drop(['sum_biomass_ug_ml', 'group_num'], axis=1).columns, predict_fn=np.exp)

fp_res_mse = calc_mse_fp(fluor_test_df, with_group_5=False)

all_results_mse = pd.concat([fp_res_mse, cleaned_res_mse]).reset_index(drop=True)

temp_res_mse = all_results_mse.rename(columns={k: figure_titles[str(k)] for k in all_results_mse.columns if k != 'Model'})

temp_res_mse['Mean'] = temp_res_mse.mean(axis=1)
temp_res_mse['std'] = temp_res_mse.std(axis=1)

def bar_plot_all_models_cleaned_no_smogn_mse(df):
    selected_columns = [2, 3, 4, 5, 6]
    selected_models = ['xgb', 'svr', 'rf', 'FP']
    filtered_df = df[df['Model'].isin(selected_models)]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    plt.xticks(rotation=90)
    
    bar_width = 0.15
    index = np.arange(len(selected_columns))

    colors = ['orange', 'blue', 'green', 'red']
    for i, (model, color) in enumerate(zip(selected_models, colors)):
        model_data = filtered_df[filtered_df['Model'] == model]
        if not model_data.empty:
            y = model_data[selected_columns].values[0] 
            ax.bar(index + i * bar_width, y, bar_width, label=model, color=color)

    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_xticks(index + bar_width * (len(selected_models) - 1) / 2)
    ax.set_xticklabels([figure_titles.get(str(x), str(x)) for x in selected_columns])
    
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

bar_plot_all_models_cleaned_no_smogn_mse(all_results_mse)

########### SVR ##########################
merged_test['predictions'] = float('inf')

for group_num in merged_test.group_num:
    group_data = merged_test[merged_test.group_num == group_num].drop(['year', 'Depth', 'week', 'group_num', 'predictions'], axis=1)
    predictions = regression_models['svr'][group_num].predict(group_data.drop(columns=['sum_biomass_ug_ml']))
    merged_test.loc[merged_test['group_num'] == group_num, 'predictions'] = np.exp(predictions)
    
def plot_combined_correlation_heatmap_depth(predictions_df, figure_titles):
    heatmap_data = predictions_df.pivot(index='Depth', columns='group_num', values='correlation').reset_index()
    heatmap_data.rename(columns={2: figure_titles['2'], 3: figure_titles['3'], 4: figure_titles['4'],
                                6: figure_titles['6']}, inplace=True)
    heatmap_data.reset_index(drop=True, inplace=True)
    
    heatmap_data['Depth'][heatmap_data['Depth'] == 3] = '0-3'
    heatmap_data['Depth'][heatmap_data['Depth'] == 5] = '3-5'
    heatmap_data['Depth'][heatmap_data['Depth'] == 10] = '5-10'
    heatmap_data['Depth'][heatmap_data['Depth'] == 15] = '10-15'
    heatmap_data['Depth'][heatmap_data['Depth'] == 20] = '15-20'
    heatmap_data['Depth'][heatmap_data['Depth'] == 21] = '20-43'
    
    heatmap_data.set_index('Depth', inplace=True)

    sorted_columns = sorted(heatmap_data.columns, key=lambda x: x)

    heatmap_data = heatmap_data[sorted_columns]
    
    plt.figure(figsize=(10, 8), dpi=300)  

    # Plot the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 12})     
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.ylabel('Depth', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.show()

correlation_matrices = []

for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for depth_value in merged_test[merged_test['group_num'] == group_num]['Depth'].unique():
        group_depth_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['Depth'] == depth_value)]
        correlation = group_depth_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]
        correlation_matrices.append({
            'group_num': group_num,
            'Depth': depth_value,
            'correlation': correlation
        })
        
correlation_df = pd.DataFrame(correlation_matrices)

print("AVG corr depth SVR = " + str(correlation_df["correlation"].mean()))
print("std corr depth SVR = " + str(correlation_df["correlation"].std()))

correlation_df_depth_svr = correlation_df.copy()
plot_combined_correlation_heatmap_depth(correlation_df, figure_titles)

def plot_combined_correlation_heatmap_month(predictions_df, figure_titles):
    heatmap_data = predictions_df.pivot(index='month', columns='group_num', values='correlation').reset_index()
    heatmap_data.rename(columns={2: figure_titles['2'], 3: figure_titles['3'], 4: figure_titles['4'],
                                6: figure_titles['6']}, inplace=True)
    heatmap_data.reset_index(drop=True, inplace=True)
    heatmap_data.set_index('month', inplace=True)
    sorted_columns = sorted(heatmap_data.columns, key=lambda x: x)

    heatmap_data = heatmap_data[sorted_columns]
    
    plt.figure(figsize=(10, 8), dpi=300)  

    ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 12})  
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.ylabel('Month', fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.show()

correlation_matrices = []

for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for month_value in merged_test[merged_test['group_num'] == group_num]['month'].unique():
        group_month_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['month'] == month_value)]
        correlation = group_month_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]
        correlation_matrices.append({
            'group_num': group_num,
            'month': month_value,
            'correlation': correlation
        })

correlation_df = pd.DataFrame(correlation_matrices)

print("AVG corr SVR month = " + str(correlation_df["correlation"].mean()))
print("std corr SVR month = " + str(correlation_df["correlation"].std()))

correlation_df_month_svr = correlation_df.copy()
plot_combined_correlation_heatmap_month(correlation_df, figure_titles)

#XGB
merged_test['predictions'] = float('inf')

for group_num in merged_test.group_num:
    group_data = merged_test[merged_test.group_num == group_num].drop(['year', 'Depth', 'week', 'group_num', 'predictions'], axis=1)
    predictions = regression_models['xgb'][group_num].predict(group_data.drop(columns=['sum_biomass_ug_ml']))
    merged_test.loc[merged_test['group_num'] == group_num, 'predictions'] = np.exp(predictions)

correlation_matrices = []
for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for depth_value in merged_test[merged_test['group_num'] == group_num]['Depth'].unique():
        group_depth_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['Depth'] == depth_value)]
        correlation = group_depth_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]

        correlation_matrices.append({
            'group_num': group_num,
            'Depth': depth_value,
            'correlation': correlation
        })
        
correlation_df = pd.DataFrame(correlation_matrices)

print("AVG corr XGB depth = " + str(correlation_df["correlation"].mean()))
print("std corr XGB depth = " + str(correlation_df["correlation"].std()))

correlation_df_depth_xgb = correlation_df.copy()
plot_combined_correlation_heatmap_depth(correlation_df, figure_titles)

correlation_matrices = []
for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for month_value in merged_test[merged_test['group_num'] == group_num]['month'].unique():
        group_month_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['month'] == month_value)]
        correlation = group_month_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]
        correlation_matrices.append({
            'group_num': group_num,
            'month': month_value,
            'correlation': correlation
        })

correlation_df = pd.DataFrame(correlation_matrices)

print("AVG corr XGB month = " + str(correlation_df["correlation"].mean()))
print("std corr XGB month = " + str(correlation_df["correlation"].std()))

correlation_df_month_xgb = correlation_df.copy()

plot_combined_correlation_heatmap_month(correlation_df, figure_titles)


#RF
merged_test['predictions'] = float('inf')

for group_num in merged_test.group_num:
    group_data = merged_test[merged_test.group_num == group_num].drop(['year', 'Depth', 'week', 'group_num', 'predictions'], axis=1)
    predictions = regression_models['rf'][group_num].predict(group_data.drop(columns=['sum_biomass_ug_ml']))
    merged_test.loc[merged_test['group_num'] == group_num, 'predictions'] = np.exp(predictions)
    
    
correlation_matrices = []

for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for depth_value in merged_test[merged_test['group_num'] == group_num]['Depth'].unique():
        group_depth_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['Depth'] == depth_value)]
        correlation = group_depth_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]
        correlation_matrices.append({
            'group_num': group_num,
            'Depth': depth_value,
            'correlation': correlation
        })
        
correlation_df = pd.DataFrame(correlation_matrices)
correlation_df.fillna(0, inplace=True)

print("AVG corr RF depth = " + str(correlation_df["correlation"].mean()))
print("std corr RF depth = " + str(correlation_df["correlation"].std()))

correlation_df_depth_rf = correlation_df.copy()
plot_combined_correlation_heatmap_depth(correlation_df, figure_titles)

correlation_matrices = []
for group_num in merged_test['group_num'].unique():
    if group_num == 5:
        continue
    for month_value in merged_test[merged_test['group_num'] == group_num]['month'].unique():
        group_month_data = merged_test[(merged_test['group_num'] == group_num) & (merged_test['month'] == month_value)]
        correlation = group_month_data[['predictions', 'sum_biomass_ug_ml']].corr().iloc[0, 1]

        correlation_matrices.append({
            'group_num': group_num,
            'month': month_value,
            'correlation': correlation
        })

correlation_df = pd.DataFrame(correlation_matrices)
correlation_df.fillna(0, inplace=True)

print("AVG corr RF month = " + str(correlation_df["correlation"].mean()))
print("std corr RF month = " + str(correlation_df["correlation"].std()))

correlation_df_month_rf = correlation_df.copy()
plot_combined_correlation_heatmap_month(correlation_df, figure_titles)


############################################################# paired t-tests

# t-test to find the highest significant group
from scipy import stats
import pandas as pd
from itertools import combinations

# Month
df1 = correlation_df_month_svr
df2 = correlation_df_month_xgb
df3 = correlation_df_month_rf

correlation_results_month_FP_df = pd.DataFrame.from_dict(correlation_results_month_FP, orient='index')
correlation_results_month_FP_df = correlation_results_month_FP_df.reset_index()
correlation_results_month_FP_df = pd.melt(correlation_results_month_FP_df, id_vars=['index'], value_vars=list(correlation_results_month_FP_df.columns[1:]), var_name='group_num', value_name='correlation')
df4 = correlation_results_month_FP_df.dropna()
df4["month"] = df4["group_num"].values.tolist()

dfs_months = [df1, df2, df3, df4]

model_names = ['SVR', 'XGB', 'RF', 'FP']

t_test_results_month = []

# Perform t-tests between each pair of models
for (df1_, model1), (df2_, model2) in combinations(zip(dfs_months, model_names), 2):
    t_statistic, p_value = stats.ttest_ind(df1_['correlation'], df2_['correlation'])
    t_test_results_month.append({
        'Comparison': f'{model1} vs {model2}',
        'T-Statistic': t_statistic,
        'P-Value': p_value
    })

results_df_month = pd.DataFrame(t_test_results_month)

# =============================================================================
# plt.figure(figsize=(10, 6))
# sns.heatmap(results_df_month.pivot(index=None, columns='Comparison', values='P-Value'),
#             annot=True, cmap='coolwarm', linewidths=.5, fmt=".3f")
# plt.title('P-Values of T-Tests between Model Pairs (Month)')
# plt.xlabel('Model Comparison')
# plt.ylabel('')  
# plt.tight_layout()
# plt.show()
# =============================================================================

combined_df = pd.concat([df1.assign(Model='SVR'), 
                         df2.assign(Model='XGB'), 
                         df3.assign(Model='RF'), 
                         df4.assign(Model='FP')])


model_order = ['SVR', 'XGB', 'RF', 'FP']

summary_stats = combined_df.groupby('Model')['correlation'].agg(['mean', 'std']).reindex(model_order)

plt.figure(figsize=(10, 6))
sns.swarmplot(x='Model', y='correlation', data=combined_df, order=model_order)

for i, model in enumerate(model_order):
    plt.errorbar(i, summary_stats.loc[model, 'mean'], yerr=summary_stats.loc[model, 'std'], 
                 fmt='none', color='black', capsize=5)

plt.scatter(range(len(model_order)), summary_stats['mean'], color='black', marker='o', s=100)

plt.title('Correlation Distribution across Models (Month)')
plt.xlabel('Model')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()


#DEPTH
df1_depth = correlation_df_depth_svr
df2_depth = correlation_df_depth_xgb
df3_depth = correlation_df_depth_rf

correlation_results_depth_FP_df = pd.DataFrame.from_dict(correlation_results_depth_FP, orient='index')
correlation_results_depth_FP_df = correlation_results_depth_FP_df.reset_index()
correlation_results_depth_FP_df = pd.melt(correlation_results_depth_FP_df, id_vars=['index'], value_vars=list(correlation_results_depth_FP_df.columns[1:]), var_name='group_num', value_name='correlation')
correlation_results_depth_FP_df["Depth"] = correlation_results_depth_FP_df["group_num"].values.tolist()
correlation_results_depth_FP_df["Depth"] = correlation_results_depth_FP_df["Depth"].replace("0-3", 1).replace("3-5", 3).replace("5-10", 5).replace("10-15", 10).replace("15-20", 15).replace("20-43", 25)
df4_depth = correlation_results_depth_FP_df.dropna()

dfs_depth = [df1_depth, df2_depth, df3_depth, df4_depth]

model_names = ['SVR', 'XGB', 'RF', 'FP']

t_test_results_depth = []

for (df1, model1), (df2, model2) in combinations(zip(dfs_depth, model_names), 2):
    t_statistic, p_value = stats.ttest_ind(df1['correlation'], df2['correlation'])
    t_test_results_depth.append({
        'Comparison': f'{model1} vs {model2}',
        'T-Statistic': t_statistic,
        'P-Value': p_value
    })

results_df_depth = pd.DataFrame(t_test_results_depth)
# =============================================================================
# 
# plt.figure(figsize=(10, 6))
# sns.heatmap(results_df_depth.pivot(index=None, columns='Comparison', values='P-Value'),
#             annot=True, cmap='coolwarm', linewidths=.5, fmt=".3f")
# plt.title('P-Values of T-Tests between Model Pairs (Depth)')
# plt.xlabel('Model Comparison')
# plt.ylabel('')  
# plt.tight_layout()
# plt.show()
# =============================================================================

combined_df = pd.concat([df1_depth.assign(Model='SVR'), 
                         df2_depth.assign(Model='XGB'), 
                         df3_depth.assign(Model='RF'), 
                         df4_depth.assign(Model='FP')])


model_order = ['SVR', 'XGB', 'RF', 'FP']

summary_stats = combined_df.groupby('Model')['correlation'].agg(['mean', 'std']).reindex(model_order)



####FIGURE    ###############################################3
# Create swarmplot
plt.figure(figsize=(10, 6))
sns.swarmplot(x='Model', y='correlation', data=combined_df, order=model_order)

for i, model in enumerate(model_order):
    plt.errorbar(i, summary_stats.loc[model, 'mean'], yerr=summary_stats.loc[model, 'std'], 
                 fmt='none', color='black', capsize=5)

plt.scatter(range(len(model_order)), summary_stats['mean'], color='black', marker='o', s=100)

plt.title('Correlation Distribution across Models (Depth)')
plt.xlabel('Model')
plt.ylabel('Correlation')
plt.tight_layout()
plt.show()


############U-TEST ###############################3

from scipy.stats import mannwhitneyu

# Define depth subgroups
depth_groups = {
    '0-15': (0, 15),
    '15-20': (15, 21),
    '<20': (21, 30)
}

mannwhitney_results_depth = []

# Perform Mann-Whitney U Test for depth subgroups
for depth_label, (min_depth, max_depth) in depth_groups.items():
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            df1 = globals()[f"df{i+1}_depth"]
            df2 = globals()[f"df{j+1}_depth"]
            
            df1_subset = df1[(df1['Depth'] >= min_depth) & (df1['Depth'] <= max_depth)]
            df2_subset = df2[(df2['Depth'] >= min_depth) & (df2['Depth'] <= max_depth)]
            
            statistic, p_value = mannwhitneyu(df1_subset['correlation'], df2_subset['correlation'])
            
            mannwhitney_results_depth.append({
                'Comparison': f'{model1} vs {model2} ({depth_label})',
                'Statistic': statistic,
                'P-Value': p_value
            })

mannwhitney_results_depth_df = pd.DataFrame(mannwhitney_results_depth)

print(mannwhitney_results_depth_df)



#DEPTH 
combined_df_depthwise = pd.concat([
    df.assign(Model=model_name) 
    for df, model_name in zip([df1_depth, df2_depth, df3_depth, df4_depth], model_names)
])

depth_groups = {
    '0-15': (0, 15),
    '15-20': (15, 21),
    '<20': (21, 30)
}

mannwhitney_results_depthwise = []

for depth_label, (min_depth, max_depth) in depth_groups.items():
    for i in range(len(depth_groups)):
        for j in range(i+1, len(depth_groups)):
            depth_label_1, depth_range_1 = list(depth_groups.items())[i]
            depth_label_2, depth_range_2 = list(depth_groups.items())[j]
            
            df1_subset = combined_df_depthwise[(combined_df_depthwise['Depth'] >= depth_range_1[0]) & (combined_df_depthwise['Depth'] < depth_range_1[1])]
            df2_subset = combined_df_depthwise[(combined_df_depthwise['Depth'] >= depth_range_2[0]) & (combined_df_depthwise['Depth'] < depth_range_2[1])]
            
            statistic, p_value = mannwhitneyu(df1_subset['correlation'], df2_subset['correlation'])
            
            mannwhitney_results_depthwise.append({
                'Comparison': f'{depth_label_1} vs {depth_label_2}',
                'Statistic': statistic,
                'P-Value': p_value
            })

mannwhitney_results_depthwise_df = pd.DataFrame(mannwhitney_results_depthwise)

print(mannwhitney_results_depthwise_df)


# =============================================================================
# 
# 
# month_groups = {
#     'Warm Season': [5, 6, 7, 8, 9, 10],
#     'Cold Season': [11, 12, 1, 2, 3, 4]
# }
# 
# mannwhitney_results_month = []
# 
# for month_label, months in month_groups.items():
#     for i in range(len(model_names)):
#         for j in range(i+1, len(model_names)):
#             model1 = model_names[i]
#             model2 = model_names[j]
#             
#             df1 = globals()[f"df{i+1}"]
#             df2 = globals()[f"df{j+1}"]
#             
#             df1_subset = df1[df1['month'].isin(months)]
#             df2_subset = df2[df2['month'].isin(months)]
#             
#             statistic, p_value = mannwhitneyu(df1_subset['correlation'], df2_subset['correlation'])
#             
#             mannwhitney_results_month.append({
#                 'Comparison': f'{model1} vs {model2} ({month_label})',
#                 'Statistic': statistic,
#                 'P-Value': p_value
#             })
# 
# mannwhitney_results_month_df = pd.DataFrame(mannwhitney_results_month)
# 
# print(mannwhitney_results_month_df)
# 
# 
# combined_df_monthwise = pd.concat([
#     df.assign(Model=model_name) 
#     for df, model_name in zip([df1, df2, df3, df4], model_names)
# ])
# 
# month_groups = {
#     'Warm Season': [5, 6, 7, 8, 9, 10],
#     'Cold Season': [11, 12, 1, 2, 3, 4]
# }
# 
# 
# mannwhitney_results_monthwise = []
# 
# for month_label_1, month_list_1 in month_groups.items():
#     for month_label_2, month_list_2 in month_groups.items():
#         if month_label_1 != month_label_2:
#             for month in month_list_1:
#                 for other_month in month_list_2:
#                     if month != other_month:
#                         df1_subset = combined_df_monthwise[combined_df_monthwise['month'] == month]
#                         df2_subset = combined_df_monthwise[combined_df_monthwise['month'] == other_month]
# 
#                         statistic, p_value = mannwhitneyu(df1_subset['correlation'], df2_subset['correlation'])
# 
#                         mannwhitney_results_monthwise.append({
#                             'Comparison': f'{month_label_1} vs {month_label_2} (Month {month} vs Month {other_month})',
#                             'Statistic': statistic,
#                             'P-Value': p_value
#                         })
# 
# mannwhitney_results_monthwise_df = pd.DataFrame(mannwhitney_results_monthwise)
# 
# print(mannwhitney_results_monthwise_df[mannwhitney_results_monthwise_df["P-Value"] <= 0.05])
# 
# 
# =============================================================================




#######################################################


####SHAP

shap_values_list_rf = plot_shap_values(merged_train, regression_models['rf'], merged_test.drop(['Depth', 'week', 'year', 'predictions'], axis=1))

shap_values_list_xgboost = plot_shap_values(merged_train, regression_models['xgb'], merged_test.drop(['Depth', 'week', 'year', 'predictions'], axis=1))

shap_values_list_svr = plot_shap_values(merged_train, regression_models['svr'], merged_test.drop(['Depth', 'week', 'year', 'predictions'], axis=1), do_sample=True)

with open(BaseFolder+'shap_values_list_xgboost.pkl', 'wb') as f:
    pickle.dump(shap_values_list_xgboost, f)
    
with open(BaseFolder+'shap_values_list_svr.pkl', 'wb') as f:
    pickle.dump(shap_values_list_svr, f)
    
with open(BaseFolder+'shap_values_list_rf.pkl', 'wb') as f:
    pickle.dump(shap_values_list_rf, f)



with open(BaseFolder+'shap_values_list_xgboost.pkl', 'rb') as f:
    shap_values_list_xgboost = pickle.load(f)
    
with open(BaseFolder+'shap_values_list_svr.pkl', 'rb') as f:
    shap_values_list_svr = pickle.load(f)
    
with open(BaseFolder+'shap_values_list_rf.pkl', 'rb') as f:
    shap_values_list_rf = pickle.load(f)


def shap_heatmap(shap_values_by_group, do_abs=True, model='SVR'):
    features_list = merged_test.drop(['group_num', 'sum_biomass_ug_ml', 'week', 'year', 'Depth', 'predictions'], axis=1).columns.tolist()
    shap_df = pd.DataFrame({k: np.average(np.abs(v.values) if do_abs else v.values, axis=0)
                            for k, v in shap_values_by_group.items()}, index=features_list)

    scaler = MinMaxScaler()
    scaled_shap_df = pd.DataFrame(scaler.fit_transform(shap_df), columns=shap_df.columns, index=shap_df.index).sort_index(axis=1)
    scaled_shap_df.rename(columns={x: figure_titles[str(x)] for x in scaled_shap_df.columns}, inplace=True)
    scaled_shap_df = scaled_shap_df.transpose().rename(columns={x: figure_titles[str(x)] for x in scaled_shap_df.transpose().columns})
    scaled_shap_df = scaled_shap_df.reindex(sorted(scaled_shap_df.columns), axis=1)

    plt.figure(figsize=(8, 6), dpi=300)
    heatmap = sns.heatmap(scaled_shap_df, cmap='coolwarm', annot=True, fmt=".1f")
    
    plt.ylabel("Taxonomic Groups", fontsize=12)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=45, fontsize=12)
    plt.show()


shap_heatmap(shap_values_list_svr, do_abs=True)

shap_heatmap(shap_values_list_xgboost, do_abs=True)

shap_heatmap(shap_values_list_rf, do_abs=True)









