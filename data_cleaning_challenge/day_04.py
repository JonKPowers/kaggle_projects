# Day 4: Character encoding issues

import pandas as pd
import numpy as np
import chardet          # a character encoding module

# Set seed
np.random.seed(101)

# Work out the character encoding
with open ('data/ks-projects-201801.csv', 'rb') as rawdata:
    result_kickstarter = chardet.detect(rawdata.read(10000))
print(result_kickstarter)

with open('data/PoliceKillingsUS.csv', 'rb') as rawdata:
    result_police = chardet.detect(rawdata.read(50000))
print(result_police)

# read in the files with their new-found encodings
kickstarter_data = pd.read_csv('data/ks-projects-201612.csv', encoding=result_kickstarter['encoding'])
police_data = pd.read_csv('data/PoliceKillingsUS.csv', encoding=result_police['encoding'])

# Save file as UTF-8
kickstarter_data.to_csv('ks-projects-201801-utf8.csv')
police_data.to_csv('PoliceKillingsUS-utf8.csv')



