import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from with_tests.evaluator import get_predictions
from with_tests.feature_transformer import extract_title, extract_gender, generate_age_estimate, \
    convert_age_guess_to_age_category, extract_family_size, extract_is_alone_indicator, calculate_age_class_combo, \
    extract_embarked_port_category
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

    full_data_df['Fare'] = full_data_df['Fare'].fillna(full_data_df['Fare'].dropna().median())

    full_data_df['FareBand'] = pd.qcut(full_data_df['Fare'], 4)

    full_data_df.loc[full_data_df['Fare'] <= 7.91, 'Fare'] = 0
    full_data_df.loc[(full_data_df['Fare'] > 7.91) & (full_data_df['Fare'] <= 14.454), 'Fare'] = 1
    full_data_df.loc[(full_data_df['Fare'] > 14.454) & (full_data_df['Fare'] <= 31), 'Fare'] = 2
    full_data_df.loc[full_data_df['Fare'] > 31, 'Fare'] = 3
    full_data_df['Fare'] = full_data_df['Fare'].astype(int)

    train_df = full_data_df[-full_data_df['Survived'].isna()]
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
        'Fare',
        # 'FareBand'
    ]]
    train_columns = train_df.columns

    X_train = train_df.drop('Survived', axis=1).copy()
    Y_train = train_df["Survived"]

    test_df = full_data_df[full_data_df['Survived'].isna()]
    test_df = test_df[list(train_df.columns.values)+["PassengerId"]]
    test_df = test_df.drop('Survived', axis=1)
    test_columns = test_df.columns
    X_test = test_df.drop("PassengerId", axis=1).copy()

    models_list = [(SVC, 'Support Vector Machines', None),
                   (KNeighborsClassifier, 'KNN', {"n_neighbors": 3}),
                   (GaussianNB, 'Naive Bayes', None),
                   (Perceptron, 'Perceptron', None),
                   (SGDClassifier, 'Stochastic Gradient Decent', None),
                   (DecisionTreeClassifier, 'Decision Tree', None),
                   (RandomForestClassifier, 'Random Forest', {"n_estimators": 100})]

    models_scores, trained_models = train_multi_models(X_train, Y_train, models_list)

    passenger_survival_predictions = get_predictions(trained_models['Random Forest'],
                                                     X_test,
                                                     test_df['PassengerId'])

    return passenger_survival_predictions, models_scores, train_columns, test_columns


if __name__ == "__main__":
    outputs, scores = run_pipeline()
    # outputs.to_csv('scored_known_passengers_titanic.csv', index=False)
