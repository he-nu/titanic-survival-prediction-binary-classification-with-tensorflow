"""Hello and welcome to the datapreparation module!

This module allows data from the titanic dataset to be
processed so it is ready for training models.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"


def title_parser(name):
    """
    Algorithm for parsing the first title in the Name column.
    """

    for i, c in enumerate(name):
        # Assuming all names have titles and titles have the character '.' at the end
        # --->
        if c == ".":
            for n, c2 in enumerate(name[:i][::-1]):
                # <---
                if c2 == " ":
                    n_index = i - n
                    return name[n_index:i]
    return None

def dummify(data, column_name:str):
    df_encoded = pd.get_dummies(data, columns=[column_name], prefix_sep='_', 
                                prefix=column_name, dtype='int')
    return df_encoded


def clean(path_to_file):

    df = pd.read_csv(path_to_file)

    df['NameLength'] = (df['Name'].apply(lambda x: len(x)) - 
                        df['Name'].apply(lambda x: len(title_parser(x))))

    sex_hash = {
    "male": 0,
    "female": 1
    }

    df['Sex'].replace(sex_hash, inplace=True)
    df['Title'] = df['Name'].apply(lambda x: title_parser(x))

    # Assuming ages correlate with titles
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))
    
    
    columns_to_dummify = ["Title", "Embarked"]
    for column in columns_to_dummify:
       df = dummify(data=df, column_name=column)

    df['HasCabin'] = df["Cabin"].apply(lambda x: 1 if type(x) == str else 0)

    df = df.drop(columns=['PassengerId', 'Ticket', 'Name', 'Cabin'])

    scaler = MinMaxScaler()
    df['NameLength'] = scaler.fit_transform(df[['NameLength']])
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Fare'] = scaler.fit_transform(df[['Fare']])

    return df


def unify():
    """
    Making sure both datasets have the same columns in the same order, 
    as the titel parser may reveal differing values for the columns. 
    """
    df_training = clean(path_to_file=TRAIN_DATA_PATH) 
    df_test = clean(path_to_file=TEST_DATA_PATH)

    missing_columns = set(df_training.columns) - set(df_test.columns)
    for col in missing_columns:
        df_test[col] = 0
    
    df_training = df_training.reindex(sorted(df_training.columns), axis=1)
    df_test = df_test.reindex(sorted(df_test.columns), axis=1)

    extra_cols_test = set(df_test.columns) - set(df_training.columns)
    df_test = df_test.drop(columns=extra_cols_test)
    df_test = df_test.drop(columns="Survived")

    return df_training, df_test

# Cleaned and scaled data.
CLEANED_TRAINING_DATA = unify()[0]
CLEANED_TEST_DATA = unify()[1]

if __name__ == "__main__":
    print(CLEANED_TRAINING_DATA.head(1).T)

    print("----------------")
    print(CLEANED_TEST_DATA.head(1).T)

    
