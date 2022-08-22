import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

#dictonary to encode the origin column as 
origin_dic = {1: 'India', 2: 'USA', 3 : 'Germany'}

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

#normalizing the numeric columns and encoding the categorical ones
def transform_columns(df):
    numeric_columns = df.select_dtypes(['float64','int64']).columns
    category_columns = ['Origin']
    
    transformer = ColumnTransformer([
                    ('standard_scalar',StandardScaler(), list(numeric_columns)),
                    ('one_hot_encoding',OneHotEncoder(),category_columns)
                ])
    return transformer.fit_transform(df)

def predict_mpg(data, model):
    
    if type(data) == dict:
        df = pd.DataFrame(data)
    else:
        df = data
        
    prepared_data = prepare_data(df)
    model_input = transform_columns(prepared_data)
    
    return model.predict(model_input)