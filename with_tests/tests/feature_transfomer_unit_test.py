import numpy as np
import pandas as pd

from with_tests.feature_transformer import extract_title


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
