import pandas as pd
import numpy as np
import pickle
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# # Load the data
data_folder = "/Users/dominguez/Documents/Kaggle_G_Research_Crypto/data/"
train = pd.read_csv(data_folder + "train.csv")
asset_details = pd.read_csv(data_folder + 'asset_details.csv')
example_test = pd.read_csv(data_folder + 'example_test.csv')

# Create separate groups for each asset id/name
for i in range(len(asset_details)):
    current_group_number = asset_details.Asset_ID[i]
    current_asset_name = asset_details.Asset_Name[i]
    current_group = train[train["Asset_ID"] == current_group_number].set_index("timestamp")
    
    current_group.dropna(axis=0, inplace=True)
    current_group = current_group.reindex(range(current_group.index[0], current_group.index[-1] + 60, 60), method='pad')
    with open(f"./data/{current_asset_name}_train.pickle", "wb") as f:
        pickle.dump(current_group, f)
        
    # print(f"Asset_ID: {current_id}, Asset_Name: {current_asset_name}")

# Now let's do the same for test sets      
for i in range(len(asset_details)):
    current_group_number = asset_details.Asset_ID[i]
    current_asset_name = asset_details.Asset_Name[i]
    current_group = example_test[example_test["Asset_ID"] == current_group_number].set_index("timestamp")
    with open(f"./data/{current_asset_name}_test.pickle", "wb") as f:
        pickle.dump(current_group, f)
    
    # print(f"Asset_ID: {i}, Asset_Name: {asset_details.Asset_Name[i]}")

# Ensure consistency in distribution of newly created datasets
for i in range(len(asset_details)):
    current_group_number = asset_details.Asset_ID[i]
    current_asset_name = asset_details.Asset_Name[i]
    
    with open(f"./data/{current_asset_name}_train.pickle", 'rb') as f:
        current_dataset = pickle.load(f)
        
    comparison = train[train["Asset_ID"] == current_group_number].set_index("timestamp")

    # print(eth.info())  
    # print(comparison.info())
    n_steps_to_correlate = 500000
    first_apple = current_dataset.Close.values[-n_steps_to_correlate:]
    second_apple = comparison.Close.values[-n_steps_to_correlate:]

    assert len(first_apple) == len(second_apple), "The input lengths do not match"

    print(f"Calculating correlations for: {current_asset_name}")
    
    corr, _ = pearsonr(first_apple, second_apple)
    corr, _ = spearmanr(first_apple, second_apple)
    print('Pearsons: %.3f' % corr, 'Spearmans: %.3f \n' % corr)