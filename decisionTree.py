from math import log
import math
import numpy as np


class DecisionTree():
	def __init__(self):
		pass

	def calInfoEnt(self, allY: np.ndarray):
		y_list = allY.tolist()
		count = {}
		for y in y_list:
			if y not in count:
				count[y] = 0
			count[y] += 1
		num = len(y_list)
		ans = -sum([count[y] / num * log(count[y] / num, 2) for y in count])
		return ans


	def chooseBestFeature(self, X, Y, featList):
		baseEntropy = self.calInfoEnt(Y)
		bestInfoGain = -math.inf
		bestFeature = -1
		for i in featList:
			featValue = set(X[:,i])
			newEntropy = 0
			for f in featValue:
				subY = Y[X[:,i]==f]
				newEntropy += self.calInfoEnt(subY) * subY.shape[0] / Y.shape[0]
			infoGain = baseEntropy - newEntropy
			if (infoGain > bestInfoGain):
				bestInfoGain = infoGain
				bestFeature = i
		return bestFeature

	def mostClass(self, allY: np.ndarray):
		y_list = allY.tolist()
		count = {}
		for y in y_list:
			if y not in count:
				count[y] = 0
			count[y] += 1
		most, mosty = -1, -1
		for y in count:
			if count[y] > most:
				most = count[y]
				mosty = y
		assert( mosty != -1 )
		return mosty


	def treeGenerate(self, subX: np.ndarray, subY: np.ndarray, featList):
		#print(subY.shape)
		if len(set(subY.tolist())) == 1:
			return int(subY[0])
		if len(featList) == 0:
			return int(self.mostClass(subY))
		bestFeat = self.chooseBestFeature(subX, subY, featList)
		myTree = {'default': int(self.mostClass(subY))}
		nextFeatList = featList.copy()
		nextFeatList.remove(bestFeat)
		featValue = set(subX[:,bestFeat])
		for f in featValue:
			subsubX = subX[subX[:,bestFeat] == f,:]
			subsubY = subY[subX[:,bestFeat] == f]
			myTree[f] = self.treeGenerate(subsubX, subsubY, nextFeatList)
		return (bestFeat, myTree)

	def fit(self, X, Y):
		if type(X) == list:
			X = np.array(X)
		if type(Y) == list:
			Y = np.array(Y)
		self.tree = self.treeGenerate(X, Y, list(range(X.shape[1])))

	def predict(self, X):
		y_pred = []
		for ins in X:
			if type(ins) != list:
				ins = ins.tolist()
			node = self.tree
			while True:
				if ins[node[0]] not in node[1]:
					newnode = node[1]['default']
				else:
					newnode = node[1][ins[node[0]]]
				if type(newnode) == int:
					y_pred.append(newnode)
					break
				else:
					node = newnode
		return y_pred

