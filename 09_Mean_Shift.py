# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import numpy as np
from MyModules.MachineLearning import handle_non_numerical_data
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''



df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
# Dropping whitenoise information
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan
for i in range(len(X)):
	original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}

for i in range(n_clusters_):
	temp_df = original_df[(original_df['cluster_group'] == float(i))]
	survival_cluster = temp_df[(temp_df['survived'] == 1)]
	survival_rate = len(survival_cluster) / len(temp_df)
	survival_rates[i] = survival_rate
print(survival_rates)
print(original_df[(original_df['cluster_group'] == 0)].describe())
print(survival_rates)
print(original_df[(original_df['cluster_group'] == 1)].describe())
print(survival_rates)
print(original_df[(original_df['cluster_group'] == 2)].describe())
print(survival_rates)
print(original_df[(original_df['cluster_group'] == 3)].describe())


# correct = 0
# for i in range(len(X)):
# 	predict_me = np.array(X[i].astype(float))
# 	predict_me = predict_me.reshape(-1, len(predict_me))
# 	prediction = clf.predict(predict_me)
# 	if prediction[0] == y[i]:
# 		correct += 1
# print('Accuracy:', correct / len(X))
