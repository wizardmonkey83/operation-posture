import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("data.csv")

x = raw_data
y = raw_data
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_training_data, y_training_data)

predictions = model.predict(x_test_data)
