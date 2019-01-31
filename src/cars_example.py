# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:51:16 2019

@author: NB23864
"""

import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image  
import pydotplus
from sklearn.externals.six import StringIO 

project_dir = os.path.dirname(os.getcwd())
path_data = os.path.join(project_dir, 'data', 'cars_dataset.csv')

df_data = pd.read_csv(path_data, sep=';', dtype={'cars':str, 'doors':str, 'seats':str})
df_data.set_index('cars', inplace=True)
df_data.head()

inputs = df_data[['buying', 'maint', 'doors', 'lugg_boot', 'safety']] # 'lugg_boot', 'safety'
targets = df_data[['target']]
inputs = pd.get_dummies(inputs)

clf = DecisionTreeClassifier(random_state=0, criterion='entropy')
clf.fit(inputs, targets)

y_pred=clf.predict(inputs)

export_graphviz(clf,out_file=os.path.join(project_dir, 'tree2.dot'))

dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())