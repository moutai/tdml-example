import numpy as np


def extract_title(input_df):
    df = input_df[['Name']].copy()
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col',
         'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping)

    df['Title'] = df['Title'].fillna(0)
    return df['Title']


def extract_gender(input_df):
    df = input_df[['Sex']].copy()
    gender = df['Sex'].map(
        {'female': 1, 'male': 0}
    ).astype(int)
    return gender


def generate_age_estimate(input_df):
    df = input_df[['Gender', 'Pclass', 'Age']].copy()
    guess_ages = np.zeros((2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[
                (df['Gender'] == i) & (df['Pclass'] == j + 1)
                ]['Age']

            guess_df = guess_df.dropna()
            age_guess = guess_df.median()

            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[
                (df['Age'].isnull())
                & (df['Gender'] == i)
                & (df['Pclass'] == j + 1),
                'Age'] = guess_ages[i, j]
    df['Age'] = df['Age'].astype(int)
    return df['Age']


def extract_family_size(input_df):
    df = input_df[['SibSp', 'Parch']].copy()
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df[['FamilySize']]


def extract_is_alone_indicator(input_df):
    df = input_df[['FamilySize']].copy()
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df[['IsAlone']]


def convert_age_guess_to_age_category(input_df):
    df = input_df[['AgeGuess']].copy()
    df['AgeCategory'] = 0
    df.loc[df['AgeGuess'] <= 16, 'AgeCategory'] = 0
    df.loc[(df['AgeGuess'] > 16) & (df['AgeGuess'] <= 32), 'AgeCategory'] = 1
    df.loc[(df['AgeGuess'] > 32) & (df['AgeGuess'] <= 48), 'AgeCategory'] = 2
    df.loc[(df['AgeGuess'] > 48) & (df['AgeGuess'] <= 64), 'AgeCategory'] = 3
    df.loc[df['AgeGuess'] > 64, 'AgeCategory'] = 4
    return df[['AgeCategory']]


def calculate_age_class_combo(full_data_df):
    return full_data_df['AgeGuess'] * full_data_df['Pclass']


def extract_embarked_port_category(input_df):
    df = input_df[['Embarked']].copy()
    freq_port = df['Embarked'].dropna().mode()[0]
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    df['EmbarkedPortCategory'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['EmbarkedPortCategory'] = df['EmbarkedPortCategory'].astype(int)
    return df[['EmbarkedPortCategory']]


def extract_fare_category(input_df):
    df = input_df[['Fare']].copy()
    df['FareGuess'] = 0
    df['FareGuess'] = df['Fare'].fillna(df['Fare'].dropna().median())
    df['FareCategory'] = 0
    df.loc[df['FareGuess'] <= 7.91, 'FareCategory'] = 0
    df.loc[(df['FareGuess'] > 7.91) & (df['Fare'] <= 14.454), 'FareCategory'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'FareCategory'] = 2
    df.loc[df['Fare'] > 31, 'FareCategory'] = 3
    df['FareCategory'] = df['FareCategory'].astype(int)
    return df[['FareCategory']]
