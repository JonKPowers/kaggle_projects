import pandas as pd
import numpy as np

from scipy import stats # for Box-Cox transformation
from mlxtend.preprocessing import minmax_scaling # for min_max scaling

import seaborn as sns
import matplotlib.pyplot as plt

# Read in data
kickstarters_2017 = pd.read_csv("data/ks-projects-201801.csv")

# Set seed
np.random.seed(101)

##########
# Made-up example on scaling
##########
# Pull 1000 data points from exponential distribution
original_data = np.random.exponential(size=1000)
scaled_data = minmax_scaling(original_data, columns=[0])    # min-max scaling
# Plot for comparison
fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title('Original Data')
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title('Scaled Data')
# plt.show()

##########
# Made-up example on normalizing
##########
normalized_data = stats.boxcox(original_data)   #Box-Cox Transformation to normalize

fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title('Original Data')
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title('Normalized Data')
# plt.show()

##########
# Kickstarter data: Scaling
##########

goal_amounts = kickstarters_2017['usd_goal_real']
kickstarter_data_scaled = minmax_scaling(goal_amounts, columns=[0])

# Plot to compare the original and scaled data
fig, ax = plt.subplots(1, 2)
sns.distplot(goal_amounts, ax=ax[0])
ax[0].set_title('Original Data')
sns.distplot(kickstarter_data_scaled, ax=ax[1])
ax[1].set_title('Scaled Data')
# plt.show()

##########
# Kickstarted data: Normalizing
##########
positive_pledges_index = kickstarters_2017['usd_pledged_real'] > 0 # For Box-Cox Transformation
positive_pledges = kickstarters_2017['usd_pledged_real'].loc[positive_pledges_index]
pledges_normalized = stats.boxcox(positive_pledges)[0]

fig, ax = plt.subplots(1, 2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title('Original Data')
sns.distplot(pledges_normalized, ax=ax[1])
ax[1].set_title('Normalized Data')
plt.show()