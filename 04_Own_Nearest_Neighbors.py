import pandas as pd
import random
from MyModules import MachineLearning as ML
NN = ML.Nearest_Neighbors()

accuracies = []

for i in range(25):
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?', -99999, inplace=True)
	df.drop(['id'], 1, inplace=True)
	full_data = df.astype(float).values.tolist()
	random.shuffle(full_data)

	test_size = 0.4
	train_set = {2: [], 4: []}
	test_set = {2: [], 4: []}
	train_data = full_data[: -int(test_size * len(full_data))]
	test_data = full_data[-int(test_size * len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			vote, confidence = NN.k_nearest_neighbors(train_set, data, k=5)
			if group == vote:
				correct += 1
			total += 1

	# print('Accuracy:', correct / total)
	accuracies.append(correct / total)

print(sum(accuracies) / len(accuracies))
