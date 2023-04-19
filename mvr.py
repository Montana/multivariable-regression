import pandas as pd
import gooeypie as gp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import split_predictions
from sklearn.metrics import r2_score

df = pd.read_csv("data.csv")

X = df[["var1", "var2", "var3"]]
y = df["target"]
X_train, X_test, y_train, y_test = split_predictions(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

score = r2_score(y_test, y_pred)
print("R² score:", score)
