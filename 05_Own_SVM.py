import numpy as np
from MyModules import MachineLearningTraning as ML
svm = ML.Support_Vector_Machine()

data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
			1: np.array([[5, 1], [6, -1], [7, 3]])}

svm.fit(data=data_dict)

predict_us = [[0, 10],
				[1, 3],
				[3, 4],
				[3, 5],
				[5, 5],
				[5, 6],
				[6, -5],
				[5, 8]]


for p in predict_us:
	svm.predict(p)

svm.visualize(data_dict)
