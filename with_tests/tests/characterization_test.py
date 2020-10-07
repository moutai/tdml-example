from with_tests.testable_titanic_ml_pipeline import run_pipeline


def test_functional_when_running_existing_code_the_pipeline_should_produce_known_output():
    output, models_scores, train_columns, test_columns = run_pipeline()
    assert set(output.columns.values) == {'PassengerId', 'Survived'}
    assert output['Survived'].mean() > 0.35
    assert models_scores['Score'].max() > 80

    assert set(train_columns.values) == {'Survived',
                                         'Pclass',
                                         'Gender',
                                         'AgeGuess',
                                         'FareCategory',
                                         'EmbarkedPortCategory',
                                         'Title',
                                         'IsAlone',
                                         'Age*Class'}

    assert set(test_columns.values) == {'Pclass',
                                        'Gender',
                                        'AgeGuess',
                                        'FareCategory',
                                        'EmbarkedPortCategory',
                                        'Title',
                                        'IsAlone',
                                        'Age*Class',
                                        'PassengerId'}
