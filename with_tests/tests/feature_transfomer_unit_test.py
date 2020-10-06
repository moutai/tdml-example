import numpy as np
import pandas as pd

from with_tests.feature_transformer import extract_title, extract_gender, generate_age_estimate
from with_tests.testable_titanic_ml_pipeline import convert_age_guess_to_age_category, extract_is_alone_indicator, \
    extract_family_size, calculate_age_class_combo


def test_extract_title_should_return_title_column():
    test_names = [
        ("Second, Mr. First", 1),
        ("Second, Mlle. First", 2),
        ("Second, Miss. First", 2),
        ("Second, Ms. First", 2),
        ("Second, Mrs. First", 3),
        ("Second, Mme. First", 3),
        ("Second, Master. First", 4),
        ("Second, Lady. First", 5),
        ("Second, Countess. First", 5),
        ("Second, Capt. First", 5),
        ("Second, Col. First", 5),
        ("Second, Don. First", 5),
        ("Second, Dr. First", 5),
        ("First, Major. Second", 5),
        ("First, Rev. Second", 5),
        ("First, Sir. Second", 5),
        ("First, Jonkheer. Second", 5),
        ("First, Dona. Second", 5)
    ]
    names = [x[0] for x in test_names]
    df = pd.DataFrame({'Name': names})
    title_column = extract_title(df)

    assert title_column is not None
    assert len(title_column) == len(df)

    expected_titles = np.array([x[1] for x in test_names])

    assert np.array_equal(title_column, expected_titles)


def test_extract_gender_should_return_gender_column():
    test_names = [
        ("male", 0),
        ("female", 1)
    ]
    gender_names = [x[0] for x in test_names]
    df = pd.DataFrame({'Sex': gender_names})
    gender_column = extract_gender(df)
    expected_column = np.array([x[1] for x in test_names])
    assert np.array_equal(gender_column, expected_column)


def test_extract_generate_age_estimate_should_return_age_estimate_column():
    input_df = pd.DataFrame(
        {'Gender': [0, 1, 0, 1, 0, 1] + [0, 1, 0, 1, 0, 1],
         'Pclass': [1, 1, 2, 2, 3, 3] + [1, 1, 2, 2, 3, 3],
         'Age': [10, 20, 30, 40, 50, 60] + [None, None, None, None, None, None]}
    )

    actual_df = generate_age_estimate(input_df)
    expected = np.array([10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60])
    assert np.array_equal(actual_df, expected)


def test_convert_age_guess_to_age_category():
    input_df = pd.DataFrame(
        {'AgeGuess': [16, 32, 48, 64, 65]}
    )
    expected_df = pd.DataFrame(
        {'AgeCategory': [0, 1, 2, 3, 4]}
    )

    actual_df = convert_age_guess_to_age_category(input_df)
    assert expected_df.equals(actual_df)


def test_extract_family_size():
    input_df = pd.DataFrame(
        {'SibSp': [0, 1, 0, 1],
         'Parch': [0, 0, 1, 1],
         'ExpectedFamilySize': [1, 2, 2, 3]}
    )

    actual_df = extract_family_size(input_df)
    np.array_equal(input_df[['ExpectedFamilySize']].values,
                   actual_df.values)


def test_extract_is_alone_indicator():
    input_df = pd.DataFrame(
        {'FamilySize': [1, 2, 3, 4],
         'ExpectedIsAlone': [1, 0, 0, 0]}
    )

    actual_df = extract_is_alone_indicator(input_df)
    np.array_equal(input_df[['ExpectedIsAlone']].values,
                   actual_df.values)


def test_extract_age_class_combo():
    input_df = pd.DataFrame(
        {'AgeGuess': [1, 2, 3, 4],
         'Pclass': [3, 4, 3, 1],
         'ExpectedCombo': [3, 8, 9, 4]}
    )
    actual_df = calculate_age_class_combo(input_df)
    np.array_equal(input_df[['ExpectedCombo']].values,
                   actual_df.values)
