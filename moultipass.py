
# So we just run a standard Linear regression using sklearn
# Then after finding out cyclicity we get the average of 'real / predicted' for each point
# in the cycle

import numpy as np
import sklearn.linear_model as lm
import pandas as pd
import math
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

data_df = pd.read_csv('vente_maillots_de_bain.csv')

# preparing regression data
data_df["time_index"] = np.arange(1, len(data_df)+1, 1)
split_index = int(len(data_df)*0.7)
data_df_mean, data_df_std = data_df["Sales"].mean(), data_df["Sales"].std()

data_df["Sales"] = (data_df["Sales"] - data_df_mean) / data_df_std

train_data_df = data_df.iloc[ : split_index]
test_data_df = data_df.iloc[split_index : ]

x_train = train_data_df[["time_index"]]
y_train = train_data_df[['Sales']]

x_test = test_data_df[["time_index"]]
y_test = test_data_df[['Sales']]

#preparing date data
data_df['Time'].to_timestamp()
data_df['Month'] = data_df['Time'].dt.month_name()

# preparing sales data
df_mean, df_std = data_df["Sales"].mean(), data_df["Sales"].std()
data_df["Sales"] = (data_df["Sales"] - df_mean) / df_std


# Splitting train and test
split_index = int(len(data_df)*0.7)
train_df = data_df.iloc[: split_index]
test_df = data_df.iloc[split_index :]

x_train = pd.DataFrame(train_df['time_index'])
y_train = pd.DataFrame(train_df['Sales'])

x_test = pd.DataFrame(test_df['time_index'])
y_test = pd.DataFrame(test_df['Sales'])

model = lm.LinearRegression()

def fit_model(x_train, y_train):
    model.fit(x_train, x_test)
    return model

def test_model(model, x_test, y_test, df_mean, df_std):
	y_test_predicted = model.predict(x_test).squeeze()  # return a * x + b
	y_test = y_test["Sales"].values
	rmse = math.sqrt(sum((y_test- y_test_predicted)**2/len(x_test))) # donnée normalisée
	print(rmse)
	
	y_test_predicted = y_test_predicted*df_std + df_mean
	y_test = y_test*df_std+df_mean	
	

	plt.figure(figsize=(15, 6))
	
	plt.plot(x_test["time_index"], y_test, "b-") # donnée originale
	plt.plot(x_test["time_index"], y_test_predicted, "r-")
	
	plt.show()
      
fit_model(x_train, y_train)
test_model(model, x_test, y_test, df_mean, df_std)