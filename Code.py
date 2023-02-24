
## Importing the Dependecies 
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score



## Data Colection and Preprocessing

data = pd.read_csv("/kaggle/input/gold-price-data/gld_price_data.csv")




## Splitting the Data

x = data.drop(["Date","GLD"],axis=1)

y = data["GLD"]

xtn,xtt,ytn,ytt = train_test_split(x,y, test_size=0.2, random_state=2, )




## Training The Model

model = RandomForestRegressor()

model.fit(xtn,ytn)




## Model Evaluation Through r2 Score 

y_pred = model.predict(xtt)

r2score = r2_score(ytt,y_pred)

r2score

print(f"The r2score of RandomForestRegressor is {r2score} ")
