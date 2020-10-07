import numpy as np
import pandas as pd

from with_tests.feature_transformer import extract_title, extract_gender, generate_age_estimate


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
