from flask import Flask, abort, redirect, url_for, render_template,send_file
from joblib import dump, load
import re
import numpy as np
import pandas as pd

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
    try:
        blue_fighter, red_fighter, weightclass, rounds = params.split(',')
        name, percent = predict(
            data, 
            model, 
            blue_fighter.replace('_', ' '), 
            red_fighter.replace('_', ' '), 
            weightclass, 
            rounds, 
            True)
    except:
        return redirect(url_for('bad_request'))
    return f'The predicted winner is <b>{name}</b> with a probability of <b>{percent}%</b>'

from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename

import os
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    file = FileField()

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    form = MyForm()
    if form.validate_on_submit():
        f = form.file.data
        filename = form.name.data + '.txt'
        #f.save(os.path.join(filename))

        df = pd.read_csv(f, header=None, on_bad_lines='skip')
        with open(filename, 'w+') as f:

            for i in range(len(df.index)):
                name, percent = predict(
                    data, 
                    model, 
                    df.iloc[i][0].strip(),
                    df.iloc[i][1].strip(),
                    df.iloc[i][2].strip(),
                    df.iloc[i][3],
                    True)
                result = f'The predicted winner is <b>{name}</b> with a probability of <b>{percent}%</b>'
                f.write(result + '\n')
            


        #return f'file {filename} uploaded'
        return send_file(filename,
                            mimetype='text/csv',
                            attachment_filename=filename,
                            as_attachment=True)
    return render_template('submit.html', form=form)

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

@app.route('/badrequest400')
def bad_request():
    return abort(400)