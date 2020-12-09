import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from decisionTree import DecisionTree
from sklearn.model_selection import KFold


def read(datapath):
	fin = open(datapath, 'r')
	data = []
	label2id = {'loss': 0, 'draw': 1, 'win': 2}
	p2f = {'b': 0, 'x': 1, 'o': 2}
	for line in fin:
		line = line.strip().split(',')
		label = label2id[line[-1]]
		feature = np.array([p2f[p] for p in line[:-1]])
		data.append((feature, label))
	fin.close()
	return data

data = read('connect-4.data')
x = np.array([d[0] for d in data])
y = np.array([d[1] for d in data])
#y = label_binarize(y, classes=list(range(3)))
kf = KFold(5, True)
all_f1 = []
for train_index, test_index in kf.split(x):
	x_train, y_train, x_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
	
	#x_train, x_test, y_train, y_test = train_test_split(x, y)
	#print('training')
	#model = OneVsRestClassifier(SVC(kernel='rbf'))
	model = DecisionTree()
	model.fit(x_train, y_train)
	#print('testing')
	y_pred = model.predict(x_test)
	all_f1.append(f1_score(y_test, y_pred, average='macro'))
print(sum(all_f1) / 5)
