import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

data = pd.read_csv('data/data.csv')
data.head()

limit_date = '2001-04-01'
#data = data[(data['date'] > limit_date)]
print(data.shape)

print("Total NaN in dataframe :" , data.isna().sum().sum())
print("Total NaN in each column of the dataframe")
na = []
for index, col in enumerate(data):
    na.append((index, data[col].isna().sum())) 
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True) 

for i in range(len(data.columns)):
    print(data.columns[na_sorted[i][0]],":", na_sorted[i][1], "NaN")


print("Total NaN in dataframe :" , data.isna().sum().sum())
print("Total NaN in each column of the dataframe")
na = []
for index, col in enumerate(data):
    na.append((index, data[col].isna().sum())) 
na_sorted = na.copy()
na_sorted.sort(key = lambda x: x[1], reverse = True) 

for i in range(len(data.columns)):
    print(data.columns[na_sorted[i][0]],":", na_sorted[i][1], "NaN")

from sklearn.impute import SimpleImputer

imp_features = ['R_Weight_lbs', 'R_Height_cms', 'B_Height_cms', 'R_age', 'B_age', 'R_Reach_cms', 'B_Reach_cms']
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

for feature in imp_features:
    imp_feature = imp_median.fit_transform(data[feature].values.reshape(-1,1))
    data[feature] = imp_feature

imp_stance = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_R_stance = imp_stance.fit_transform(data['R_Stance'].values.reshape(-1,1))
imp_B_stance = imp_stance.fit_transform(data['B_Stance'].values.reshape(-1,1))
data['R_Stance'] = imp_R_stance
data['B_Stance'] = imp_B_stance

na_features = ['B_avg_BODY_att', 'R_avg_BODY_att']
data.dropna(subset = na_features, inplace = True)

data.drop(['Referee', 'location'], axis = 1, inplace = True)
data.drop(['B_draw', 'R_draw'], axis=1, inplace=True)

data = data[data['Winner'] != 'Draw']
data = data[data['weight_class'] != 'Catch Weight']

#  i = index of the fighter's fight, 0 means the last fight, -1 means first fight
def select_fight_row(data, name, i): 
    data_temp = data[(data['R_fighter'] == name) | (data['B_fighter'] == name)]  # filter data on fighter's name
    data_temp.reset_index(drop=True, inplace=True) #  as we created a new temporary dataframe, we have to reset indexes
    idx = max(data_temp.index)  #  get the index of the oldest fight
    if i > idx:  #  if we are looking for a fight that didn't exist, we return nothing
        return 
    arr = data_temp.iloc[i,:].values
    return arr

select_fight_row(data, 'Amanda Nunes', 0) #  we get the last fight of Amanda Nunes

# get all active UFC fighters (according to the limit_date parameter)
def list_fighters(data, limit_date):
    data_temp = data[data['date'] > limit_date]
    set_R = set(data_temp['R_fighter'])
    set_B = set(data_temp['B_fighter'])
    fighters = list(set_R.union(set_B))
    return fighters

fighters = list_fighters(data, '2017-01-01')
print(len(fighters))

def build_data(data, fighters, i):      
    arr = [select_fight_row(data, fighters[f], i) for f in range(len(fighters)) if select_fight_row(data, fighters[f], i) is not None]
    cols = [col for col in data] 
    data_fights = pd.DataFrame(data=arr, columns=cols)
    data_fights.drop_duplicates(inplace=True)
    data_fights['title_bout'] = data_fights['title_bout'].replace({True: 1, False: 0})
    data_fights.drop(['R_fighter', 'B_fighter', 'date'], axis=1, inplace=True)
    return data_fights

data_train = build_data(data, fighters, 0)
data_test = build_data(data, fighters, 1)

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer((OrdinalEncoder(), ['weight_class', 'B_Stance', 'R_Stance']), remainder='passthrough')

# If the winner is from the Red corner, Winner label will be encoded as 1, otherwise it will be 0 (Blue corner)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data_train['Winner'])
y_test = label_encoder.transform(data_test['Winner'])

X_train, X_test = data_train.drop(['Winner'], axis=1), data_test.drop(['Winner'], axis=1)

random_forest = RandomForestClassifier(n_estimators=130, 
                                       criterion='entropy', 
                                       max_depth=10, 
                                       min_samples_split=2,
                                       min_samples_leaf=1, 
                                       random_state=0)

model_forest = Pipeline([('encoding', preprocessor), ('random_forest', random_forest)])
model_forest.fit(X_train, y_train)

# We use cross-validation with 7-folds to have a more precise accuracy (reduce variation)
accuracies = cross_val_score(estimator=model_forest, X=X_train, y=y_train, cv=7)
print('Accuracy mean : ', accuracies.mean())
print('Accuracy standard deviation : ', accuracies.std())

y_pred = model_forest.predict(X_test)
print('Testing accuracy : ', accuracy_score(y_test, y_pred), '\n')

target_names = ["Blue","Red"]
print(classification_report(y_test, y_pred, labels=[0,1], target_names=target_names))


from joblib import dump
dump(model_forest, 'model_forest.joblib') 