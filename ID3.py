"""
Make the imports of python packages needed
"""
import pandas as pd
import numpy as np
from pprint import pprint
from sklearn import metrics
eps = np.finfo(float).eps
from numpy import log2 as log


#load dataset
df = pd.read_csv('Weathersmall6.csv')
df=df.drop('Location',axis=1)
df=df.drop('Date',axis=1)
#df=df.drop('RISK_MM',axis=1)

columns = df.columns.values.tolist()
dataset = df.dropna()

def train_test_split(dataset):
    train_data = dataset.iloc[80:].reset_index(drop=True)#We drop the index respectively relabel the index
    #starting form 0, because we do not want to run into errors regarding the row labels / indexes
    test_data = dataset.iloc[:80].reset_index(drop=True)
    return train_data,test_data

train_data = train_test_split(dataset)[0]
test_data = train_test_split(dataset)[1]

ent = 0
values = df.RainTomorrow.unique()
for value in values:
    fraction = df.RainTomorrow.value_counts()[value] / len(df.RainTomorrow)
    ent += -fraction * np.log2(fraction)

def ent(data, attribute):
    target_variables = data.RainTomorrow.unique()
    variables = data[attribute].unique()

    entropy_attribute = 0
    for variable in variables:
        entropy_each_feature = 0
        for target_variable in target_variables:
            num = len(data[attribute][data[attribute] == variable][data.RainTomorrow == target_variable])  # numerator
            den = len(data[attribute][data[attribute] == variable])
            fraction = num / (den + eps)  # pi
            entropy_each_feature += -fraction * log(
                fraction + eps)
        fraction2 = den / len(data)
        entropy_attribute += -fraction2 * entropy_each_feature

    return (abs(entropy_attribute))

a_entropy = {k:ent(train_data,k) for k in train_data.keys()[:-1]}
a_entropy

def ig(e_dataset,e_attr):
    return(e_dataset-e_attr)

IG = {k:ig(ent,a_entropy[k]) for k in a_entropy}
IG

def find_entropy(data):
    Class = data.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = data[Class].unique()
    for value in values:
        fraction = data[Class].value_counts()[value]/len(data[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

def find_entropy_attribute(data,attribute):
    Class = data.keys()[-1]
    target_variables = data[Class].unique()
    variables = data[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(data[attribute][data[attribute]==variable][data[Class] ==target_variable])
            den = len(data[attribute][data[attribute]==variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(data)
        entropy2 += -fraction2*entropy
    return abs(entropy2)

def find_winner(data):
    Entropy_att = []
    IG = []
    for key in columns[:-1]:
        IG.append(find_entropy(data)-find_entropy_attribute(data,key))
    return data.keys()[:-1][np.argmax(IG)] #columns[:-1][np.argmax(IG)]#


def ID3(data, originaldata, features, target_attribute_name="RainTomorrow", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        best_feature = find_winner(data)
        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            tree[best_feature][value] = subtree

        return (tree)

def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default

            result = tree[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)

            else:
                return result

def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predict = pd.DataFrame(columns=["predict"])
    for i in range(len(data)):
        predict.loc[i, "predict"] = predict(queries[i], tree, 1.0)
    print('Accuracy : ', (np.sum(predict["predict"] == data.RainTomorrow) / len(data)) * 100, '%')

tree = ID3(train_data,train_data,train_data.columns[:-1])

pprint(tree)
test(test_data,tree)