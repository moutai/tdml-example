import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from with_tests.evaluator import get_predictions
from with_tests.feature_transformer import extract_title, extract_gender, generate_age_estimate, \
    convert_age_guess_to_age_category, extract_family_size, extract_is_alone_indicator, calculate_age_class_combo, \
    extract_embarked_port_category, extract_fare_category
from with_tests.model_trainer import train_multi_models


def get_datasets_folder():
    project_directory = os.environ.get("PROJECT_DIR", '..')
    datasets_folder = f"{project_directory}/datasets/titanic/"
    return datasets_folder


def load_data(datasets_folder):
    train_df = pd.read_csv(datasets_folder + "0_train_data.csv")
    test_df = pd.read_csv(datasets_folder + "1_known_data_to_score.csv")
    df = pd.concat([train_df, test_df])
    return df


def run_pipeline():
    datasets_folder = get_datasets_folder()
    full_data_df = load_data(datasets_folder)

    full_data_df['Title'] = extract_title(full_data_df)

    full_data_df['Gender'] = extract_gender(full_data_df)

    full_data_df['AgeGuess'] = generate_age_estimate(full_data_df)

    full_data_df['AgeCategory'] = convert_age_guess_to_age_category(full_data_df)

    full_data_df['FamilySize'] = extract_family_size(full_data_df)

    full_data_df['IsAlone'] = extract_is_alone_indicator(full_data_df)

    full_data_df['Age*Class'] = calculate_age_class_combo(full_data_df)

    full_data_df['EmbarkedPortCategory'] = extract_embarked_port_category(full_data_df)

    full_data_df['FareCategory'] = extract_fare_category(full_data_df)

    X_full, Y_full, X_test, train_columns, test_columns, test_df = get_model_ready_train_and_test_sets(full_data_df)

    models_list = [(SVC, 'Support Vector Machines', None),
                   (KNeighborsClassifier, 'KNN', {"n_neighbors": 3}),
                   (GaussianNB, 'Naive Bayes', None),
                   (Perceptron, 'Perceptron', None),
                   (SGDClassifier, 'Stochastic Gradient Decent', None),
                   (DecisionTreeClassifier, 'Decision Tree', None),
                   (RandomForestClassifier, 'Random Forest', {"n_estimators": 100})]

    X_train, X_validation, y_train, y_validation = train_test_split(X_full,
                                                                    Y_full,
                                                                    test_size=0.2,
                                                                    random_state=0)

    models_scores, trained_models = train_multi_models(X_train, y_train,
                                                       X_validation, y_validation,
                                                       models_list)

    passenger_survival_predictions = get_predictions(trained_models['Random Forest'],
                                                     X_test,
                                                     test_df['PassengerId'])

    return passenger_survival_predictions, models_scores, train_columns, test_columns


def get_model_ready_train_and_test_sets(input_df):
    df = input_df.copy()
    train_df = df[-df['Survived'].isna()]
    train_df = train_df[[
        'Survived',
        'Title',
        'Pclass',
        'Gender',
        'AgeGuess',
        # 'AgeCategory',
        'IsAlone',
        # 'FamilySize'
        'Age*Class',
        'EmbarkedPortCategory',
        'FareCategory',
    ]]
    train_columns = train_df.columns
    X_train = train_df.drop('Survived', axis=1).copy()
    Y_train = train_df["Survived"]

    test_df = input_df[input_df['Survived'].isna()]
    test_df = test_df[list(train_df.columns.values) + ["PassengerId"]]
    test_df = test_df.drop('Survived', axis=1)
    test_columns = test_df.columns
    X_test = test_df.drop("PassengerId", axis=1).copy()

    return X_train, Y_train, X_test, train_columns, test_columns, test_df


if __name__ == "__main__":
    outputs, scores, _, _ = run_pipeline()
    # outputs.to_csv('scored_known_passengers_titanic.csv', index=False)