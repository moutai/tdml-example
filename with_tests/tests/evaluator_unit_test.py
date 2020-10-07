from unittest.mock import MagicMock

from with_tests.evaluator import get_predictions


def test_get_predictions_should_return_labelled_data():
    trained_models = MagicMock()
    trained_models.predict.return_value = [1, 1, 1]
    X_test = [[0, 0], [1, 1], [2, 1]]
    passenger_ids = [1, 2, 3]

    passenger_survival_predictions = get_predictions(trained_models,
                                                     X_test,
                                                     passenger_ids)
    assert len(passenger_survival_predictions) == len(passenger_ids)
    assert passenger_survival_predictions['Survived'].mean() == 1
