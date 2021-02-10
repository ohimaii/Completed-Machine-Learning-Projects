import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle as pkl
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''This function loads the data we will be working with
    Args:
    database_filepath - Filepath with previously created database containing
                        our table of interest
    Outputs:
    X - Array containing values of the independent variable
    Y - Array containing labels for each X
    category_names - names of each category to be used in classifier
    '''
    # Create engine to connect database to program
    engine = create_engine('sqlite:///' + database_filepath)
    
    table_name = os.path.basename(database_filepath).replace(".db", "") + "_table"
    
    # Read SQL table that exists in the provided database file
    df = pd.read_sql_table(table_name, con = engine)
    
    # Assign values to variables
    X = df['message'].values
    Y = np.array(df.iloc[:, 4:])
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names

def find_replace_urls(text):
    ''' This function takes in text, searches for urls 
    using the specified url regex, and replaces the urls with a placeholder'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    return text

def normalize(text):
    '''This function normalizes text by case and removes
    punctuation'''
    text = re.sub(r"[^a-z0-9A-Z]", " ", find_replace_urls(text).lower().strip())
    return text

def clean_tokenize(text):
    '''This function tokenizes text, lemmatizes it AND removes stopwords. 
    Text passed into this function should already be
    case and punctuation normalized'''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # tokenize
    tokens = word_tokenize(normalize(text))
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

    
def build_model(X_train, Y_train, classifier):
    '''This function builds a model pipeline for the selected classifier
    Output - Classification Pipeline
    '''
    scaler = StandardScaler(with_mean = False)
    vect = CountVectorizer(tokenizer = clean_tokenize)
    tfidf = TfidfTransformer()
    clf = MultiOutputClassifier(classifier(random_state = 5))
    # Build Pipeline
    pipeline = Pipeline([   
                    ('vect', vect),
                    ('tfidf', tfidf),
                    ('scaler', scaler),
                    ('clf', clf)])
    #define parameters for GridSearchCV
    parameters = {'clf__estimator__n_estimators': [50, 70, 90]}

    #create gridsearch object and return as final model pipeline
    cv_object = GridSearchCV(estimator = pipeline, 
                        param_grid = parameters, 
                        scoring = 'accuracy', cv = 3, n_jobs = -1)
    cv_object.fit(X_train, Y_train)
    best_parameters = cv_object.best_params_

    print("The best parameters are: {}".format(best_parameters))
    
    final_pipeline = Pipeline([   
                    ('vect', vect),
                    ('tfidf', tfidf),
                    ('scaler', scaler),
                    ('clf', MultiOutputClassifier(
                        classifier(n_estimators = best_parameters['clf__estimator__n_estimators'],
                                   random_state = 5)))])
    return final_pipeline


def display_results(model, X_test, Y_test, category_names):
    '''This function displays classification results for each
    category
    Outputs - 1. Precision, Recall, F1-Score, Support for each category
              2. Overall accuracy score for the classification
    '''
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    # Print the classification report for each label
    for i in range(len(category_names)):
        print('Category: {}'.format(category_names[i]))
        print(classification_report((Y_test[:,i]), (Y_pred[:,i])))
    accuracy = (Y_pred == Y_test).mean()
    print('Model Accuracy {:.3f}'.format(accuracy))
    

def save_model(model, model_filepath):
    '''This function saves the model as a pickle (.pkl) file'''
    pkl.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train, AdaBoostClassifier)
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        display_results(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()