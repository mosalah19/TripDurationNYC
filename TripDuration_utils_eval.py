from sklearn.metrics import mean_squared_error, r2_score


def predict_eval(model, train, train_features, name, model_name):
    y_train_pred = model.predict(train)
    rmse = mean_squared_error(train_features, y_train_pred, squared=False)
    r2 = r2_score(train_features, y_train_pred)
    print(model_name+"  : ")
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")
