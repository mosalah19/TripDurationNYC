from TripDuration_Data_utils import *
from TripDuration_utils_eval import *
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import pandas as pd


def Ridge_with_polynomial(train_x, train_y, val_x, val_y, degree=2, intercipt=True):

    pipeline = Pipeline(steps=[
        ('preprocessor', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regression', Ridge(alpha=1, fit_intercept=intercipt))
    ])

    model = pipeline.fit(train_x, train_y)

    predict_eval(model, train_x, train_y, "train",
                 model_name='Ridge_with_polynomial')
    predict_eval(model, val_x, val_y, "test",
                 model_name='Ridge_with_polynomial')
    return model


def simple_Ridge(train_x, train_y, val_x, val_y, intercipt=True):

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', LinearRegression(fit_intercept=intercipt))
    ])

    model = pipeline.fit(train_x, train_y)

    predict_eval(model, train_x, train_y, "train", model_name='simple_Ridge')
    predict_eval(model, val_x, val_y, "test", model_name='simple_Ridge')
    return model


def random_forest_regression(X_train, X_test, y_train, test_y):
    # Create and train the Random Forest regression model
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', RandomForestRegressor(
            n_estimators=25, min_samples_leaf=25, min_samples_split=25))
    ])
    model = pipeline.fit(X_train, y_train)
    predict_eval(model, train_x, train_y, "train",
                 model_name='random_forest_regression')
    predict_eval(model, X_test, test_y, "test",
                 model_name='random_forest_regression')
    return model


def xgboost_regression(X_train, X_test, y_train, test_y):
    # Create and train the XGBoost regression model
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regression', XGBRegressor(random_state=42))
    ])
    model = pipeline.fit(X_train, y_train)
    predict_eval(model, train_x, train_y, "train",
                 model_name='xgboost_regression')
    predict_eval(model, X_test, test_y, "test",
                 model_name='xgboost_regression')
    return model


# model = simple_Ridge(train_x, train_y, val_x, val_y)
if __name__ == '__main__':

    # load dataset
    df_train, df_test, df = load_train_dataset()
    df_val = load_val_dataset()
    # this column used only for compute trip duration
    df = df.drop(['dropoff_datetime'], axis=1)
    df_val = df_val.drop(['dropoff_datetime'], axis=1)
    # make preprocessing on data
    df = prepare_data(df)
    df_val = prepare_data(df_val)
    # split to data and target
    train_x = df.drop('trip_duration', axis=1)
    train_y = df['trip_duration']
    val_x = df_val.drop('trip_duration', axis=1)
    val_y = df_val['trip_duration']
    # train models and evaluate by rmse and r2_score
    model = simple_Ridge(
        train_x, train_y, val_x, val_y, intercipt=True)
    '''
    simple_Ridge  : 
    train RMSE = 0.4039 - R2 = 0.6529
    simple_Ridge  : 
    test RMSE = 0.4032 - R2 = 0.6549

    '''
    model = Ridge_with_polynomial(
        train_x, train_y, val_x, val_y, degree=2, intercipt=True)
    '''
    Ridge_with_polynomial  : 
    train RMSE = 0.3564 - R2 = 0.7298
    Ridge_with_polynomial  : 
    test RMSE = 0.3559 - R2 = 0.7311

    '''
    model = random_forest_regression(
        X_train=train_x, X_test=val_x, y_train=train_y, test_y=val_y)
    '''
    random_forest_regression  : 
    train RMSE = 0.2783 - R2 = 0.8352
    random_forest_regression  : 
    test RMSE = 0.3089 - R2 = 0.7975
    
    '''
    model = xgboost_regression(
        X_train=train_x, X_test=val_x, y_train=train_y, test_y=val_y)
    '''
    xgboost_regression  : 
    train RMSE = 0.2982 - R2 = 0.8108
    xgboost_regression  : 
    test RMSE = 0.3026 - R2 = 0.8056
    '''
