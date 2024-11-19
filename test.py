import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from TripDuration_Data_utils import *
import numpy as np
with open(r'tripDurationProduction.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train)
    rmse = mean_squared_error(train_features, y_train_pred, squared=False)
    r2 = r2_score(train_features, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


#

def test(url=r"D:\4st\1 project-nyc-taxi-trip-duration\split_sample\test.csv"):
    df_test = pd.read_csv(
        rf"{url}")
    df_test['pickup_datetime'] = pd.to_datetime(
        df_test['pickup_datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df_test = prepare_data(df_test)

    test_x = df_test.drop('trip_duration', axis=1)
    test_y = df_test['trip_duration']
    predict_eval(model, test_x, test_y, 'test')


if __name__ == '__main__':
    # add test path as parameter in test function
    test()
