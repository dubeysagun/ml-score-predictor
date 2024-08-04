import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\User\Downloads\ml score.csv")
df.sample(5)
df.info()
df['sum'] = df[['Libraries',	'Statistics',	'Basic maths',	'supervised algorithms',	'unsupervised algorithms',	'semi-supervised algorithms'	,'reinforced algorithm']].sum(axis = 1)
# Combined condition for filtering
conditions = (
    (df['sum']== 0 ) & (df['ML score'] > 0) |
    (df['sum'] == 1) & (df['ML score'] > 15) |
    (df['sum'] == 2) & (df['ML score'] < 30) & (df['ML score'] > 20) |
    (df['sum'] == 3) & (df['ML score'] <= 40) & (df['ML score'] > 30) |
    (df['sum'] == 4) & (df['ML score'] > 40) & (df['ML score'] < 50) |
    (df['sum'] == 5) & (df['ML score'] > 57) & (df['ML score'] < 70)|
    (df['sum'] == 6) & (df['ML score'] > 68) |
    (df['sum'] == 7) & (df['ML score'] > 80)
)
df = df[conditions]
df.head(10)
df.info()
plt.scatter(df['sum'],df['ML score'])
from sklearn.model_selection import train_test_split
X_train,X_test , y_train , y_test = train_test_split(df.drop(columns = ['ML score','sum']),df['ML score'],test_size = 0.2)
X_train
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train , y_train)
# Predict on the test data
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluate the model using various metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

df.columns
import pickle as pickle
with open('model_pickle_new','wb') as file:
    pickle.dump(model,file)
    
# loading saved model
with open('model_pickle_new','rb') as file:
    mp = pickle.load(file)