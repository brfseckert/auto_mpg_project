import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer

#dictonary to encode the origin column
origin_dic = {1: 'India', 2: 'USA', 3 : 'Germany'}

#data for fitting the parameters
# defining the column names
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
# reading the .data file using pandas
data = pd.read_csv('./auto-mpg.data', names=cols, na_values = "?",
                comment = '\t',
                sep= " ",
                skipinitialspace=True)
#making a copy of the dataframe
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['Cylinders']):
    train_set = data.loc[train_index]

train_data = train_set.drop('MPG',axis=1)


def _encode_origin_column(df):
    df['Origin'] = df['Origin'].map(origin_dic)
    return df

def _adjust_missing_values(df):
    numeric_columns = df.select_dtypes(['float64','int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    return df

def _add_custom_variables(df):
    df = df.assign(acceleration_on_cyl=df['Acceleration']/df['Cylinders'],
                   acceleration_on_power=df['Acceleration'] / df['Horsepower'])
    return df

def prepare_data(df):
    
    df = (
        df.pipe(_encode_origin_column).
        pipe(_adjust_missing_values).
        pipe(_add_custom_variables)
    )
    
    return df

train_fit_data = prepare_data(train_data)

#normalizing the numeric columns and encoding the categorical ones
def transform_columns(df):
    numeric_columns = df.select_dtypes(['float64','int64']).columns
    category_columns = ['Origin']
    
    transformer = ColumnTransformer([
                    ('standard_scalar',StandardScaler(), list(numeric_columns)),
                    ('one_hot_encoding',OneHotEncoder(categories=[["India", "USA", "Germany"]]),category_columns)
                ])
    transformer = transformer.fit(train_fit_data)

    return transformer.transform(df)

def predict_mpg(data, model):

    print(type(data))
    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
        
    prepared_data = prepare_data(df)
    print(prepared_data)
    model_input = transform_columns(prepared_data)
    print(model_input)
    
    return model.predict(model_input)