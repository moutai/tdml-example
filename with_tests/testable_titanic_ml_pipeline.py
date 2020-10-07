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
from with_tests.feature_transformer import extract_title, extract_gender, generate_age_estimate
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

    full_data_df.loc[full_data_df['AgeGuess'] <= 16, 'AgeGuess'] = 0
    full_data_df.loc[(full_data_df['AgeGuess'] > 16) & (full_data_df['AgeGuess'] <= 32), 'AgeGuess'] = 1
    full_data_df.loc[(full_data_df['AgeGuess'] > 32) & (full_data_df['AgeGuess'] <= 48), 'AgeGuess'] = 2
    full_data_df.loc[(full_data_df['AgeGuess'] > 48) & (full_data_df['AgeGuess'] <= 64), 'AgeGuess'] = 3
    full_data_df.loc[full_data_df['AgeGuess'] > 64, 'AgeGuess'] = 4

    full_data_df['FamilySize'] = full_data_df['SibSp'] + full_data_df['Parch'] + 1

    full_data_df['IsAlone'] = 0
    full_data_df.loc[full_data_df['FamilySize'] == 1, 'IsAlone'] = 1

    full_data_df['Age*Class'] = full_data_df['AgeGuess'] * full_data_df['Pclass']
    full_data_df.loc[:, ['Age*Class', 'AgeGuess', 'Pclass']].head(10)
    freq_port = full_data_df.Embarked.dropna().mode()[0]

    full_data_df['Embarked'] = full_data_df['Embarked'].fillna(freq_port)
    full_data_df['Embarked'] = full_data_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    full_data_df['Fare'] = full_data_df['Fare'].fillna(full_data_df['Fare'].dropna().median())

    full_data_df['FareBand'] = pd.qcut(full_data_df['Fare'], 4)

    full_data_df.loc[full_data_df['Fare'] <= 7.91, 'Fare'] = 0
    full_data_df.loc[(full_data_df['Fare'] > 7.91) & (full_data_df['Fare'] <= 14.454), 'Fare'] = 1
    full_data_df.loc[(full_data_df['Fare'] > 14.454) & (full_data_df['Fare'] <= 31), 'Fare'] = 2
    full_data_df.loc[full_data_df['Fare'] > 31, 'Fare'] = 3
    full_data_df['Fare'] = full_data_df['Fare'].astype(int)

    full_data_df = full_data_df.drop(['Name',
                                      'Sex',
                                      'Age',
                                      'Ticket',
                                      'Cabin',
                                      'FareBand'], axis=1)

    train_df = full_data_df[-full_data_df['Survived'].isna()]
    train_df = train_df.drop(['PassengerId'], axis=1)
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]

    test_df = full_data_df[full_data_df['Survived'].isna()]
    test_df = test_df.drop('Survived', axis=1)
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

    return passenger_survival_predictions, models_scores


if __name__ == "__main__":
    outputs, scores = run_pipeline()
    # outputs.to_csv('scored_known_passengers_titanic.csv', index=False)
