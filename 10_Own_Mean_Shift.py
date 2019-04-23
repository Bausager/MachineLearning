import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
import random
from MyModules.MachineLearningTraning import Mean_Shift
clf = Mean_Shift()

style.use('ggplot')
colors = 10 * ["g", "r", "c", "b", "k"]

centers = random.randrange(2, 5)
X, y = make_blobs(n_samples=50, centers=centers, n_features=2)

clf.fit(X)

centroids = clf.centroids
for classification in clf.classifications:
	color = colors[classification]
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150)

for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)
plt.show()
