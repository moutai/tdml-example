from sklearn.ensemble import RandomForestClassifier

from with_tests.model_trainer import train_multi_models


def test_multi_models_train_should_return_models_and_info():
    models_list = [(RandomForestClassifier, 'Random Forest', {"n_estimators": 100})]
    X_train = [[0, 0], [1, 1]]
    Y_train = [0, 1]
    models_scores, trained_models = train_multi_models(X_train, Y_train, models_list)
    assert models_scores['Score'].mean() == 100.0
    assert len(trained_models) == 1
