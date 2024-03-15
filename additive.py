import pandas as pd
import sklearn.linear_model as lm

data_df = pd.read_csv('vente_maillots_de_bain.csv')

#preparing date data
data_df['time'].to_timestamp()
data_df['month'] = data_df['time'].dt.month_name()
one_hot_encoded_df = pd.get_dummies(data_df, columns='month')

# Preparing the sales data
split_index = int(len(one_hot_encoded_df)*0.7)
df_mean, df_std = one_hot_encoded_df["Sales"].mean(), one_hot_encoded_df["Sales"].std()

one_hot_encoded_df["Sales"] = (one_hot_encoded_df["Sales"] - df_mean) / df_std

train_df = one_hot_encoded_df.iloc[ : split_index]
test_df = one_hot_encoded_df.iloc[split_index : ]

x_train = one_hot_encoded_df['']
y_train = pd.DataFrame(train_df['Sales'])

one_hot_encoded_df.head()
x_test = test_df['']
y_test = pd.DataFrame(test_df['Sales'])

# Need to split train and test

model = lm.LinearRegression()
model.fit()