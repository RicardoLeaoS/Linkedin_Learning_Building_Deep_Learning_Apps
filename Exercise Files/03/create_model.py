import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("sales_data_training_scaled.csv")

X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# 9 features
#print(np.shape(X))

#print(training_data_df.head())

# Define the model
model = Sequential()
model.add(Dense(32, input_dim=9, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer = "adam")

#model.fit(x=X, y=Y, batch_size = 10, epochs= 100)

#model.save("seq_model_1.h5")


#test_data_df  = pd.read_csv("sales_data_testing_scaled.csv")

#X_test = test_data_df.drop('total_earnings', axis=1).values
#y_test = test_data_df[['total_earnings']].values

#model.evaluate(x=X_test, y=y_test)


# CNN
# keras.layers.convolutional.Conv2D()

# Recurrent layer: LSTM
# keras.layers.recurrent.LSTM()
