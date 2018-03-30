import pandas as pd
import numpy as np
import seaborn as sns

# Read in data
earthquakes = pd.read_csv('data/significant_earthquakes_database.csv')
landslides = pd.read_csv('data/landslides_catalog.csv')
volcanos = pd.read_csv('data/volcanic_eruptions_database.csv')

# Set seed
np.random.seed(101)

# Parse dates
landslides['date'] = pd.to_datetime(landslides['date'], format='%m/%d/%y')
earthquakes['Date'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)

# Pull day of month from each
day_of_month_landslides = landslides['date'].dt.day
day_of_month_earthquakes = earthquakes['Date'].dt.day

# Remove na's
day_of_month_landslides = day_of_month_landslides.dropna()
day_of_month_earthquakes = day_of_month_earthquakes.dropna()


# Plot day of months
sns.distplot(day_of_month_landslides, kde=False, bins=31)
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)



