"""Hello and welcome to the datapreparation module!

This module allows data from the titanic dataset to be
processed so it is ready for training models.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    df_encoded = pd.get_dummies(data, columns=[column_name], prefix_sep='_', prefix=column_name, dtype='int')
    return df_encoded

def unique_titles(path_to_file=TRAIN_DATA_PATH):
    df = pd.read_csv(path_to_file)
    unique_titles = df['Title'].unique()

    return unique_titles

def parse_ticket(ticket):
    separated = ticket.split()
    if len(separated) > 1:
        return separated[0]
    else:
        return "no_expansion"

def scale_pipeline(data, column_name):
    scaler_std = StandardScaler()
    scaler_mm = MinMaxScaler()
    data[column_name] = scaler_std.fit_transform(data[[column_name]])
    data[column_name] = scaler_mm.fit_transform(data[[column_name]])
    
    return data



def clean(path_to_file):

    df = pd.read_csv(path_to_file)

    df['NameLength'] = df['Name'].apply(lambda x: len(x)) - df['Name'].apply(lambda x: len(title_parser(x)))


    sex_hash = {
    "male": 0,
    "female": 1
    }

    df['Sex'].replace(sex_hash, inplace=True)
    df['Title'] = df['Name'].apply(lambda x: title_parser(x))

    # Assuming ages correlate with titles
    df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.mean()))
    
    unique_titles = df['Title'].unique()

    df['Ticket'] = df['Ticket'].apply(lambda x: parse_ticket(x))
    
    columns_to_dummify = ["Title", "Embarked", "Ticket"]
    for column in columns_to_dummify:
       df = dummify(data=df, column_name=column)


    df['HasCabin'] = df["Cabin"].apply(lambda x: 1 if type(x) == str else 0)
    df = df.drop(columns=['PassengerId', 'Name', 'Cabin'])


    # insert pipeline
    # columns_to_scale = ['NameLength', 'Age', 'Fare']

    # ValueError: Columns must be same length as key
    # for column_n in columns_to_scale:
    #     df[column_n] = scale_pipeline(data=df, column_name=column_n)
    
    scaler_std = StandardScaler()
    scaler_MM = MinMaxScaler()

    df['NameLength'] = scaler_std.fit_transform(df[['NameLength']])
    df['Age'] = scaler_std.fit_transform(df[['Age']])
    df['Fare'] = scaler_std.fit_transform(df[['Fare']])

    df['NameLength'] = scaler_MM.fit_transform(df[['NameLength']])
    df['Age'] = scaler_MM.fit_transform(df[['Age']])
    df['Fare'] = scaler_MM.fit_transform(df[['Fare']])


    return df

def unify():
    CLEANED_TRAINING_DATA = clean(path_to_file=TRAIN_DATA_PATH) 
    CLEANED_TEST_DATA = clean(path_to_file=TEST_DATA_PATH)

    missing_columns = set(CLEANED_TRAINING_DATA.columns) - set(CLEANED_TEST_DATA.columns)
    for col in missing_columns:
        CLEANED_TEST_DATA[col] = 0
    
    CLEANED_TRAINING_DATA = CLEANED_TRAINING_DATA.reindex(sorted(CLEANED_TRAINING_DATA.columns), axis=1)
    CLEANED_TEST_DATA = CLEANED_TEST_DATA.reindex(sorted(CLEANED_TEST_DATA.columns), axis=1)

    extra_cols_test = set(CLEANED_TEST_DATA.columns) - set(CLEANED_TRAINING_DATA.columns)
    CLEANED_TEST_DATA = CLEANED_TEST_DATA.drop(columns=extra_cols_test)
    CLEANED_TEST_DATA = CLEANED_TEST_DATA.drop(columns="Survived")


    return CLEANED_TRAINING_DATA, CLEANED_TEST_DATA

CLEANED_TRAINING_DATA = unify()[0]
CLEANED_TEST_DATA = unify()[1]

if __name__ == "__main__":
    print(CLEANED_TRAINING_DATA.head(1).T)

    print("----------------")
    print(CLEANED_TEST_DATA.head(1).T)