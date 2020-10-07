import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from with_tests.evaluator import get_model_accuracy, get_predictions
from with_tests.feature_transformer import extract_title
from with_tests.model_trainer import train_model, train_multi_models


def run_pipeline():
    datasets_folder = get_datasets_folder()
    full_data_df = load_data(datasets_folder)

    full_data_df['Title'] = extract_title(full_data_df)

    full_data_df = full_data_df.drop(['Ticket', 'Cabin'], axis=1)
    full_data_df = full_data_df.drop(['Name'], axis=1)

    full_data_df['Sex'] = full_data_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    guess_ages = np.zeros((2, 3))

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = full_data_df[(full_data_df['Sex'] == i) & (full_data_df['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            full_data_df.loc[(full_data_df.Age.isnull()) & (full_data_df.Sex == i) & (full_data_df.Pclass == j + 1), \
                   'Age'] = guess_ages[i, j]
    full_data_df['Age'] = full_data_df['Age'].astype(int)
    full_data_df.head()
    full_data_df['AgeBand'] = pd.cut(full_data_df['Age'], 5)
    full_data_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
    full_data_df.loc[full_data_df['Age'] <= 16, 'Age'] = 0
    full_data_df.loc[(full_data_df['Age'] > 16) & (full_data_df['Age'] <= 32), 'Age'] = 1
    full_data_df.loc[(full_data_df['Age'] > 32) & (full_data_df['Age'] <= 48), 'Age'] = 2
    full_data_df.loc[(full_data_df['Age'] > 48) & (full_data_df['Age'] <= 64), 'Age'] = 3
    full_data_df.loc[full_data_df['Age'] > 64, 'Age'] = 4

    full_data_df = full_data_df.drop(['AgeBand'], axis=1)

    full_data_df['FamilySize'] = full_data_df['SibSp'] + full_data_df['Parch'] + 1
    full_data_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False)
    full_data_df['IsAlone'] = 0
    full_data_df.loc[full_data_df['FamilySize'] == 1, 'IsAlone'] = 1
    full_data_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
    full_data_df = full_data_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    full_data_df['Age*Class'] = full_data_df.Age * full_data_df.Pclass
    full_data_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
    freq_port = full_data_df.Embarked.dropna().mode()[0]

    full_data_df['Embarked'] = full_data_df['Embarked'].fillna(freq_port)
    full_data_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                          ascending=False)
    full_data_df['Embarked'] = full_data_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    full_data_df['Fare'] = full_data_df['Fare'].fillna(full_data_df['Fare'].dropna().median())

    full_data_df['FareBand'] = pd.qcut(full_data_df['Fare'], 4)
    full_data_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
    full_data_df.loc[full_data_df['Fare'] <= 7.91, 'Fare'] = 0
    full_data_df.loc[(full_data_df['Fare'] > 7.91) & (full_data_df['Fare'] <= 14.454), 'Fare'] = 1
    full_data_df.loc[(full_data_df['Fare'] > 14.454) & (full_data_df['Fare'] <= 31), 'Fare'] = 2
    full_data_df.loc[full_data_df['Fare'] > 31, 'Fare'] = 3
    full_data_df['Fare'] = full_data_df['Fare'].astype(int)
    full_data_df = full_data_df.drop(['FareBand'], axis=1)

    train_df = full_data_df[-full_data_df['Survived'].isna()]
    train_df = train_df.drop(['PassengerId'], axis=1)
    test_df = full_data_df[full_data_df['Survived'].isna()]
    test_df = test_df.drop('Survived', axis=1)

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
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


def load_data(datasets_folder):
    train_df = pd.read_csv(datasets_folder + "0_train_data.csv")
    test_df = pd.read_csv(datasets_folder + "1_known_data_to_score.csv")
    df = pd.concat([train_df, test_df])
    return df


def get_datasets_folder():
    project_directory = os.environ.get("PROJECT_DIR", '..')
    datasets_folder = f"{project_directory}/datasets/titanic/"
    return datasets_folder


if __name__ == "__main__":
    passenger_survival_predictions, models_scores = run_pipeline()
    # passenger_survival_predictions.to_csv('scored_known_passengers_titanic.csv', index=False)
