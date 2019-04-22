import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from MyModules.MachineLearning import K_Means
style.use('ggplot')
colors = 10 * ["g", "r", "c", "b", "k"]

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

clf = K_Means()
clf.fit(X)

unknowns = np.array([[1, 11],
					[8, 9],
					[0, 3],
					[5, 4],
					[6, 4]])


for centroids in clf.centroids:
	plt.scatter(clf.centroids[centroids][0], clf.centroids[centroids][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)


for unknown in unknowns:
	classification = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plt.show()
