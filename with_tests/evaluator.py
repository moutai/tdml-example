import pandas as pd


def get_model_accuracy(trained_model, X_train, Y_train):
    acc_svc = round(trained_model.score(X_train, Y_train) * 100, 2)
    return acc_svc


def get_predictions(trained_model, X_test, passenger_ids):
    passenger_survival_predictions = trained_model.predict(X_test)
    output = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": passenger_survival_predictions
    })
    return output
