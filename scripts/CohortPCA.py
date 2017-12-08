import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import random
import operator 
import math
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

#ControlPanel = "data/AncestryPCA.master.panel"
#EVEC = "data/Freeze3.index.gz.plus.HapMap.pca.evec"

class Control:
	def __init__(self, row):
		self.ID, self.Pop, self.SuperPop, self.Gender = row

class Evec:
	def __init__(self, row):
		self.ID = row[0].split(":")[-1]
		self.Partition = row[-1]
		self.EigenValues = row[1:-1]

class CohortPCA:
	def __init__(self, ControlPanel, EVEC):
		self.ControlPanelFil = ControlPanel
		self.EvecFil = EVEC
	def loadControlPanel(self):
		# sample  pop     super_pop       gender
		self.ControlPanel = {}
		fin = open(self.ControlPanelFil, 'rb')
		reader = csv.reader(fin, delimiter='\t')
		header = reader.next()
		for row in reader:
			Indv = Control(row)
			self.ControlPanel[Indv.ID] = Indv
	def loadEigenValue(self):
		fin = open(self.EvecFil, 'rb')
		self.Colors = {'SAS':"green" , 'EAS':"blue", 'AMR':"orange", 'AFR':"purple", 'EUR':"red", "AJ":"pink", "DOMI":"yellow" ,'CASE':"black"}
		self.SuperPops = {'SAS':[] , 'EAS':[], 'AMR':[], 'AFR':[], 'EUR':[], 'AJ':[], "DOMI":[]} #'SAS', 'EAS', 'AMR', 'AFR', 'EUR'
		self.Cases = []
		header = fin.next()
		for line in fin:
			row = line.strip().split()
			evec = Evec(row)
			if evec.Partition == "Case":
				self.Cases.append(evec)
			elif evec.Partition == "Control":
				try:
					superpop = self.ControlPanel[evec.ID].SuperPop
					self.SuperPops[superpop].append(evec)
				except KeyError:
					print "Sample %s marked as control but not in Control Panel" % (evec.ID)

	def ReadEigenvaluesFromList(self, List, idx1, idx2):
		X, Y = [], []
		for evec in List:
			X.append(evec.EigenValues[idx1])
			Y.append(evec.EigenValues[idx2])
		return X, Y

	def MakeTrainingSet(self):
		res = []
		for k,v in self.SuperPops.items():
			for evec in v:
				evec.SuperPop = k
				res.append(evec)
		random.shuffle(res)
		Data = np.array([x.EigenValues for x in res])
		Label = np.array([x.SuperPop for x in res])
		return Data, Label

	def MarkCases(self, Output, model="RF"):
		with open(Output, 'wb') as fout:
			writer = csv.writer(fout, delimiter="\t")
			TrainingSet, TrainingLabel = self.MakeTrainingSet()
			if model == "RF":
				model = RandomForest(TrainingSet, TrainingLabel)
			elif model == "GB":
				model = GradientBoosting(TrainingSet, TrainingLabel)
			elif model == "SVM":
				model = Support_Vector_Machine(TrainingSet, TrainingLabel)
			model.fit()
			scores = cross_val_score(model.model, TrainingSet, TrainingLabel, cv=5)
			print "CrossV:", scores
			print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
			#print model.model.classes_ 
			writer.writerow(["IndvID", "Population", "Probability", ",".join(model.model.classes_), "EigenValues"])
			for case in self.Cases:
				response = model.predict(np.array(case.EigenValues).reshape(1, -1))
				Label_response = zip(model.model.classes_, response)
				MaxP , MaxLabel = self.GetLabel(Label_response)
				response = [str(round(x,2)) for x in response]
				#fout.write("%s\t%s\t%s\n"%(case.ID, MaxLabel, MaxP,','.join(case.EigenValues)))
				writer.writerow([case.ID, MaxLabel, MaxP, ",".join(response), ",".join(case.EigenValues)])
	
	def GetLabel(self, response):
		MaxP, MaxLabel = 0, None
		for label, p in response:
			if p > MaxP:
				MaxP, MaxLabel = p, label
		return MaxP, MaxLabel

class KNN():
	def __init__(self, K=5):
		self.K = K

	def euclideanDistance(self, vector1, vector2):
		distance = 0
		for a,b in zip(vector1, vector2):
			distance += pow((float(a) - float(b)), 2)
		return math.sqrt(distance)

	def getNeighbors(self, trainingSet, testInstance):
		distances = []
		for x in trainingSet:
			dist = self.euclideanDistance(testInstance.EigenValues, x.EigenValues)
			distances.append((x, dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(self.K):
			neighbors.append(distances[x][0])
		return neighbors

	def getResponse(self, neighbors):
		classVotes = {}
		for x in range(len(neighbors)):
			response = neighbors[x].SuperPop
			if response in classVotes:
				classVotes[response] += 1
			else:
				classVotes[response] = 1
		sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

class Support_Vector_Machine:
	def __init__(self, TrainingData, TrainingLabels):
		kernel = 'rbf'
		print "Model: Support_Vector_Machine", kernel
		self.model = svm.SVC(kernel=kernel, gamma=10)
		self.trainingData = TrainingData
		self.trainingLabels = TrainingLabels
	# train
	def fit(self):
		s_time = time.time()
		print "Fitting Model..."
		self.model.fit(self.trainingData, self.trainingLabels)
		print "Done with Fitting, used %.4f secs." % (time.time() - s_time)
	def predict(self, case):
		dec = self.svm.predict_proba(case)[0]
		return dec

class RandomForest:
	def __init__(self, TrainingData, TrainingLabels):
		print "Model: RandomForestClassifier"
		self.model = RandomForestClassifier(
				n_estimators=50, 
				max_depth=None, 
				min_samples_split=20, 
				random_state=1)
		#self.model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, max_depth=None, random_state=0)
		self.trainingData = TrainingData
		self.trainingLabels = TrainingLabels
	# train
	def fit(self):
		s_time = time.time()
		print "Fitting Model..."
		self.model.fit(self.trainingData, self.trainingLabels)
		print "Done with Fitting, used %.4f secs." % (time.time() - s_time)
	def predict(self, case):
		dec = self.model.predict_proba(case)[0]
		return dec

class GradientBoosting:
	def __init__(self, TrainingData, TrainingLabels):
		print "Model: GradientBoostingClassifier"
		self.model = GradientBoostingClassifier(
				n_estimators=10, 
				learning_rate=0.5, 
				max_depth=None, 
				random_state=0)
		self.trainingData = TrainingData
		self.trainingLabels = TrainingLabels
	# train
	def fit(self):
		s_time = time.time()
		print "Fitting Model..."
		self.model.fit(self.trainingData, self.trainingLabels)
		print "Done with Fitting, used %.4f secs." % (time.time() - s_time)
	def predict(self, case):
		dec = self.model.predict_proba(case)[0]
		return dec
