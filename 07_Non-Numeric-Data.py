# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from MyModules.MachineLearning import handle_non_numerical_data

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
# Dropping whitenoise information
df.drop(['body', 'name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])
clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
	predict_me = np.array(X[i].astype(float))
	predict_me = predict_me.reshape(-1, len(predict_me))
	prediction = clf.predict(predict_me)
	if prediction[0] == y[i]:
		correct += 1
print('Accuracy:', correct / len(X))
