from flask import Flask
from joblib import dump, load
import re
import numpy as np
import pandas as pd
import requests

data = pd.read_csv('data/data.csv')
model = load('model_forest.joblib') 

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

def predict(data, pipeline, blue_fighter, red_fighter, weightclass, rounds, title_bout=False): 
    
    #We build two dataframes, one for each figther 
    f1 = data[(data['R_fighter'] == blue_fighter) | (data['B_fighter'] == blue_fighter)].copy()
    f1.reset_index(drop=True, inplace=True)
    f1 = f1[:1]
    f2 = data[(data['R_fighter'] == red_fighter) | (data['B_fighter'] == red_fighter)].copy()
    f2.reset_index(drop=True, inplace=True)
    f2 = f2[:1]
    
    # if the fighter was red/blue corner on his last fight, we filter columns to only keep his statistics (and not the other fighter)
    # then we rename columns according to the color of  the corner in the parameters using re.sub()
    if (f1.loc[0, ['R_fighter']].values[0]) == blue_fighter:
        result1 = f1.filter(regex='^R', axis=1).copy() #here we keep the red corner stats
        result1.rename(columns = lambda x: re.sub('^R','B', x), inplace=True)  #we rename it with "B_" prefix because he's in the blue_corner
    else: 
        result1 = f1.filter(regex='^B', axis=1).copy()
    if (f2.loc[0, ['R_fighter']].values[0]) == red_fighter:
        result2 = f2.filter(regex='^R', axis=1).copy()
    else:
        result2 = f2.filter(regex='^B', axis=1).copy()
        result2.rename(columns = lambda x: re.sub('^B','R', x), inplace=True)
        
    fight = pd.concat([result1, result2], axis = 1) # we concatenate the red and blue fighter dataframes (in columns)
    fight.drop(['R_fighter','B_fighter'], axis = 1, inplace = True) # we remove fighter names
    fight.insert(0, 'title_bout', title_bout) # we add tittle_bout, weight class and number of rounds data to the dataframe
    fight.insert(1, 'weight_class', weightclass)
    fight.insert(2, 'no_of_rounds', rounds)
    fight['title_bout'] = fight['title_bout'].replace({True: 1, False: 0})
    
    pred = pipeline.predict(fight)
    proba = pipeline.predict_proba(fight)
    if (pred == 1.0): 
        predict_list = [red_fighter, round(proba[0][1] * 100, 2)]
        #str_predict ="The predicted winner is", red_fighter, 'with a probability of', round(proba[0][1] * 100, 2), "%"
    else:
        predict_list = [blue_fighter, round(proba[0][1] * 100, 2)]
        str_predict ="The predicted winner is", blue_fighter, 'with a probability of ', round(proba[0][0] * 100, 2), "%"
    #return proba
    return predict_list

@app.route('/prediction_winner/<params>')
def prediction_winner(params):
    blue_fighter, red_fighter, weightclass, rounds = params.split(',')
    print(blue_fighter)
    name, percent = predict(
        data, 
        model, 
        blue_fighter.replace('_', ' '), 
        red_fighter.replace('_', ' '), 
        weightclass, 
        rounds, 
        True)
    return f'The predicted winner is <b>{name}</b> with a probability of <b>{percent}%</b>'

@app.route('/show_image/')
def show_image():
    #response = requests.get('https://www.ufc-time.ru/wp-content/uploads/2020/03/Дональд-Серроне.png', stream=True)
    #response.content
    #<img src="/static/Джон-Джонс.png" alt="lorem">
    #if response:
    #    print('Success!')
    #else:
    #    print('An error has occurred.')
    return '<img src="/static/Джон-Джонс.png" alt="lorem">'
