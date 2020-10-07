import pandas as pd

from with_tests.evaluator import get_model_accuracy


def train_model(Model, X_train, Y_train, **kwargs):
    model = Model(**kwargs)
    model.fit(X_train, Y_train)
    return model


def train_multi_models(input_X_train, input_y_train,
                       input_X_validation, input_y_validation,
                       models_list):
    trained_models = {}

    models_scores = {'Model': [],
                     'Score': []}
    for model, name, params in models_list:
        trained_model = train_model(model, input_X_train, input_y_train)
        accuracy_score = get_model_accuracy(trained_model, input_X_validation, input_y_validation)
        models_scores['Model'].append(name)
        models_scores['Score'].append(accuracy_score)
        trained_models[name] = trained_model
    models_scores = pd.DataFrame.from_dict(models_scores)
    models_scores = models_scores.sort_values(by='Score', ascending=False)
    return models_scores, trained_models
