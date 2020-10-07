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
from with_tests.model_trainer import train_model, train_multi_models


def run_pipeline():
    project_directory = os.environ.get("PROJECT_DIR", '..')
    datasets_folder = f"{project_directory}/datasets/titanic/"
    train_df = pd.read_csv(datasets_folder + "0_train_data.csv")
    test_df = pd.read_csv(datasets_folder + "1_known_data_to_score.csv")
    df = pd.concat([train_df, test_df])

    df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
    df = df.drop(['Ticket', 'Cabin'], axis=1)

    df['Title'] = df.Name.str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

    df = df.drop(['Name'], axis=1)

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)

    guess_ages = np.zeros((2, 3))

    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), \
                   'Age'] = guess_ages[i, j]
    df['Age'] = df['Age'].astype(int)
    df.head()
    df['AgeBand'] = pd.cut(df['Age'], 5)
    df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4

    df = df.drop(['AgeBand'], axis=1)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False)
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
    df = df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

    df['Age*Class'] = df.Age * df.Pclass
    df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
    freq_port = df.Embarked.dropna().mode()[0]

    df['Embarked'] = df['Embarked'].fillna(freq_port)
    df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                          ascending=False)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    df['Fare'] = df['Fare'].fillna(df['Fare'].dropna().median())

    df['FareBand'] = pd.qcut(df['Fare'], 4)
    df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    df = df.drop(['FareBand'], axis=1)

    train_df = df[-df['Survived'].isna()]
    train_df = train_df.drop(['PassengerId'], axis=1)
    test_df = df[df['Survived'].isna()]
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


if __name__ == "__main__":
    passenger_survival_predictions, models_scores = run_pipeline()
    # passenger_survival_predictions.to_csv('scored_known_passengers_titanic.csv', index=False)
