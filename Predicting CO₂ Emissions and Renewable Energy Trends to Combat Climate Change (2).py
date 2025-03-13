#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# In[13]:


# Load CO2 emissions and renewable energy datasets
co2_data = pd.read_csv("C:/Users/nauti/OneDrive/Desktop/DATA/CO2_cleaned.csv") #Replace with your file path
renewable_data = pd.read_csv("C:/Users/nauti/OneDrive/Desktop/DATA/renewable-share-energy.csv")


# In[14]:


# Merge datasets
data = pd.merge(co2_data, renewable_data, on=["Country", "Year"])

# Display the cleaned and merged dataset
data.head()


# In[31]:


# Clean and preprocess data
data['Density'] = data['Density'].astype(str)
data['Density'] = data['Density'].str.extract('(\d+)', expand=False).astype(float)
data.ffill(inplace=True)
data['Per_Capita_Emissions'] = data['CO2_Emissions'] / data['Population']
data['Renewable_Growth'] = data.groupby('Country')['Renewable_Share'].diff().fillna(0)


# In[16]:


# Normalize features
scaler = MinMaxScaler()
data[['Population', 'Density', 'Renewable_Share', 'CO2_Emissions']] = scaler.fit_transform(
    data[['Population', 'Density', 'Renewable_Share', 'CO2_Emissions']]
)


# In[17]:


# Prepare features and target
features = ['Population', 'Density', 'Renewable_Share']
target = 'CO2_Emissions'


# In[18]:


# Split dataset for a specific country (e.g., 'USA')
country_data = data[data['Country'] == 'United States']
X = country_data[features].values
y = country_data[target].values


# In[19]:


# Reshape data for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))


# In[20]:


# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# In[21]:


# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


# In[22]:


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)


# In[23]:


# Evaluate the model
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R² Score: {r2_score(y_test, y_pred)}")


# In[24]:


# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("CO2 Emissions Prediction using LSTM")
plt.show()


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Filter data and create a copy to avoid the warning
ind_data = data[data['Country'] == 'India'].copy()

# Convert 'Year' to datetime and set as index
ind_data['Date'] = pd.to_datetime(ind_data['Year'], format='%Y')
ind_data.set_index('Date', inplace=True)


# Prepare data for modeling
X = ind_data.index.year.values.reshape(-1, 1)  # Use 'Year' as the predictor
y = ind_data['CO2_Emissions'].values           # Target variable

# Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Forecast next 5 years
future_years = np.arange(X[-1][0] + 1, X[-1][0] + 6).reshape(-1, 1)  # Next 5 years
forecast = model.predict(future_years)

# Combine actual and forecast data for visualization
plt.figure(figsize=(10, 6))
plt.plot(X, y, label='Actual', marker='o')
plt.plot(future_years, forecast, label='Forecast', marker='x', linestyle='--')

# Plot formatting
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions')
plt.title('CO₂ Emissions Forecast for India')
plt.legend()
plt.grid(True)
plt.show()

# Display forecast values
forecast_df = pd.DataFrame({'Year': future_years.flatten(), 'Forecast_Emissions': forecast})
print("CO₂ Emissions Forecast:")
print(forecast_df)


# In[ ]:


# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['CO2_Emissions', 'Renewable_Share']])

# Visualize clusters
sns.scatterplot(
    data=data, x='CO2_Emissions', y='Renewable_Share', hue='Cluster', palette='viridis'
)
plt.title("Clustering of Countries")
plt.show()


# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Example setup
train_data = pd.DataFrame({
    'Year': [2000, 2001, 2002, 2003, 2004],
    'Emissions': [10, 12, 15, 20, 22]
})

# Test data
X_test = pd.DataFrame({
    'Year': [2005, 2006, 2007, 2008, 2009],
    'Emissions': [24, 25, 27, 30, 32],
})

# Scenario 1: Continuation of current trend
# Linear regression for Scenario 1
model_1 = LinearRegression()
model_1.fit(train_data[['Year']], train_data['Emissions'])
scenario_1_emissions = model_1.predict(X_test[['Year']])

# Scenario 2: Faster growth (assume a 10% annual increase in emissions)
train_data['Adjusted_Emissions'] = train_data['Emissions'] * 1.1  # Increase by 10%
model_2 = LinearRegression()
model_2.fit(train_data[['Year']], train_data['Adjusted_Emissions'])
scenario_2_emissions = model_2.predict(X_test[['Year']])

# Compare results
print("Scenario 1 Average Emissions:", np.mean(scenario_1_emissions))
print("Scenario 2 Average Emissions:", np.mean(scenario_2_emissions))


# In[3]:


import matplotlib.pyplot as plt
import numpy as np

# Mock data for demonstration
np.random.seed(42)
y_test = np.random.normal(loc=300, scale=50, size=1000)  # Actual emissions
scenario_1_emissions = y_test * 0.8  # Scenario 1: Renewable energy doubled (reduced emissions)
scenario_2_emissions = y_test * 1.1  # Scenario 2: Population increase (higher emissions)

# Visualize Scenario Impacts
plt.figure(figsize=(10, 6))
plt.hist(y_test, alpha=0.5, label='Original Emissions')
plt.hist(scenario_1_emissions, alpha=0.5, label='Scenario 1: Double Renewable')
plt.hist(scenario_2_emissions, alpha=0.5, label='Scenario 2: +10% Population')
plt.legend()
plt.title("Scenario Analysis: Emission Predictions")
plt.xlabel("CO2 Emissions")
plt.ylabel("Frequency")
plt.show()


# In[4]:


# Insights from scenarios
if np.mean(scenario_1_emissions) < np.mean(y_test):
    print("Policy Recommendation: Invest in renewable energy projects to reduce emissions.")
if np.mean(scenario_2_emissions) > np.mean(y_test):
    print("Policy Warning: Population growth without emissions management increases CO2 emissions.")


# In[ ]:




