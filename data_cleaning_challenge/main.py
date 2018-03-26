import pandas as pd
import numpy as np

# Read in data
nfl_data = pd.read_csv('data/NFL_play_by_play_2009-2016.csv')
sf_permits = pd.read_csv('data/Building_Permits.csv')

# Set random seed
np.random.seed(100)

#See how many values are missing
missing_values_count_nfl = nfl_data.isnull().sum()
total_cells_nfl = np.product(nfl_data.shape)
total_missing_nfl = missing_values_count_nfl.sum()
percent_missing_nfl = (total_missing_nfl/total_cells_nfl) * 100
print(f'Percent of NFL cells with missing data: {percent_missing_nfl}')

missing_values_count_sf = sf_permits.isnull().sum()
total_cells_sf = np.product(sf_permits.shape)
total_missing_sf = missing_values_count_sf.sum()
percent_missing_sf = (total_missing_sf/total_cells_sf) * 100
print(f'Percent of SF cells with missing data {percent_missing_sf}')

# Option 1: Drop columns with missing data
columns_with_na_dropped_nfl = nfl_data.dropna(axis=1)
print(f'Columns in original NFL dataset: {nfl_data.shape[1]}')
print(f'Columns with nas dropped: {columns_with_na_dropped_nfl.shape[1]}')

columns_with_na_dropped_sf = sf_permits.dropna(axis=1)
print(f'Columns in original SF dataset: {sf_permits.shape[1]}')
print(f'Columns with nas dropped: {columns_with_na_dropped_sf.shape[1]}')

# Option 2: Fill in missing values
nfl_data_with_imputation = nfl_data.fillna(method='bfill', axis=0).fillna('0')
print(nfl_data_with_imputation.sample(5))

sf_permits_with_imputation = sf_permits.fillna(method='bfill', axis=0).fillna('0')
print(sf_permits_with_imputation.sample(5))