import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score

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
x = [d[0] for d in data]
y = [d[1] for d in data]
y = label_binarize(y, classes=list(range(3)))
x_train, x_test, y_train, y_test = train_test_split(x, y)
print('training')
model = OneVsRestClassifier(SVC(kernel='rbf'))
model.fit(x_train, y_train)
print('testing')
y_pred = model.predict(x_test)
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='micro'))

