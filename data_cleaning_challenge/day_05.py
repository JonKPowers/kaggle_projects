# Day 5--Inconsistent Data Entry

import pandas as pd
import numpy as np
import chardet
import fuzzywuzzy
from fuzzywuzzy import process

# Set seed
np.random.seed(101)

# Work out the character encoding
with open('data/PakistanSuicideAttacks_Ver_11.csv', 'rb') as raw_data:
    result_1 = chardet.detect(raw_data.read(50000))
print(result_1)

with open('data/PakistanSuicideAttacks_Ver_6.csv', 'rb') as raw_data:
    result_2 = chardet.detect(raw_data.read(50000))
print(result_2)

# Read in the data
suicide_attacks_1 = pd.read_csv('data/PakistanSuicideAttacks_Ver_11.csv', encoding=result_1['encoding'])
suicide_attacks_2 = pd.read_csv('data/PakistanSuicideAttacks_Ver_6.csv', encoding=result_2['encoding'])

# Take a look at the unique values in the 'City' column
cities = suicide_attacks_1['City'].unique()
cities.sort()
print(cities)

# Convert to lowercase and strip leading and trailing whitespace
suicide_attacks_1['City'] = suicide_attacks_1['City'].str.lower()
suicide_attacks_1['City'] = suicide_attacks_1['City'].str.strip()
suicide_attacks_1['Province'] = suicide_attacks_1['Province'].str.lower()
suicide_attacks_1['Province'] = suicide_attacks_1['Province'].str.strip()

# Function to replace rows in the provided column of a dataframe
# that match the provided string about the provided ratio

def replace_matches_in_column(df, column, string_to_match, min_ratio=90):
    strings = df[column].unique()       # Get unique strings
    # Get 10 closest matches to input string and limit to mathces >= min_ratio
    matches = fuzzywuzzy.process.extract(string_to_match, strings,
                                         limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
    close_matches = [match[0] for match in matches if match[1] >= min_ratio]
    # Pull rows with close matches and replace with input matches
    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = string_to_match

    print('All done!')

# Replace close matches to 'd.i khan'
replace_matches_in_column(df=suicide_attacks_1, column='City', string_to_match="d.i khan")
replace_matches_in_column(df=suicide_attacks_1, column='City', string_to_match='kurram agency')






