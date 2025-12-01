import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class RFC:
	def __init__(self,n_estimators=500,
			  max_depth=None,
			  min_samples_leaf=1,
			  max_features='sqrt',
			  bootstrap=True,
			  class_weight=None,
			  random_state= RANDOM_STATE,
			  n_jobs=-1,
			  oob_score=True):
		self.n_estimators=n_estimators
		self.max_depth=max_depth
		self.min_samples_leaf=min_samples_leaf
		self.max_features=max_features
		self.bootstrap=bootstrap
		self.class_weight=class_weight
		self.random_state=random_state
		self.n_jobs=n_jobs
		self.oob_score=oob_score
		 
	def build_random_forest(self):
		rf = RandomForestClassifier(
    		n_estimators=self.n_estimators,
    		max_depth=self.max_depth,
    		min_samples_leaf=self.min_samples_leaf,
    		max_features=self.max_features,
    		bootstrap=self.bootstrap,
			class_weight=self.class_weight,
			random_state=self.random_state,
			n_jobs=self.n_jobs,
			oob_score=self.oob_score
		)
		return rf
	
	def train_model(self,model, X_train, y_train):
		model.fit(X_train, y_train)
		if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
			print(f"OOB Score: {model.oob_score_:.4f}")
		return model

class DT:
	def __init(self):
		pass

	def train_model(self,x_train,y_train):
		clf = tree.DecisionTreeClassifier()

		clf = clf.fit(x_train, y_train)

		return clf	

class CNNClassifier:
    pass
