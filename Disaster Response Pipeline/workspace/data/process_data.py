import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''This function loads the messages and categories data from their 
    respective sources and merges them into one dataframe
    
    Args:
    messages_filepath - user's file path leading to the messages csv file
    categories_filepath - user's file path leading to the categories csv file
    
    Output:
    df - combined dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on= 'id', how = 'left')
    return df

def clean_data(df):
    '''This function cleans the combined dataframe by splitting the categories in the `categories`
    column into multiple binary columns (one for each category)
    Args:
    df - combined messages and categories dataframe 
    
    Output:
    df - new dataframe with individual binary category columns
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = list(categories.iloc[0,:])

    # use this row to extract a list of new column names for categories with slicing 
    category_colnames = [category[:-2] for category in row]
    
    # rename the columns of the new categories dataframe
    categories.columns = category_colnames
    
    # convert category values to binary
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace the categories column in the dataframe with the new split and cleaned categories
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis=1)
    
    # Drop Duplicates
    df.drop_duplicates(inplace = True)
    
    # Drop of rows with related values of 2 as they do not seem to add any useful meaning.
    indexes = df[df['related'] == 2].index
    df.drop(indexes, inplace = True)
    
    # Drop the `child_alone` column because it is not a useful category for the model
    df.drop('child_alone', axis = 1, inplace = True)
    return df

def save_data(df, database_filepath):
    '''This function saves the data as an SQL table in 
    the provided database filepath
    
    Args:
    df - data to be saved
    database_filepath - filepath to the database that contains the table
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df.to_sql(table_name, engine, if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()