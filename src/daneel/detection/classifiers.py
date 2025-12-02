import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K


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

class DL:
	def build_simple_cnn(n_bins=1000):
		"""Simpler CNN to prevent overfitting on small datasets."""
		print("\n" + "="*70)
		print("BUILDING SIMPLIFIED CNN")
		print("="*70)
		
		model = models.Sequential([
			layers.Input(shape=(n_bins, 1)),
			
			# Feature extraction
			layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
			layers.BatchNormalization(),
			layers.MaxPooling1D(2),
			layers.Dropout(0.3),
			
			layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
			layers.BatchNormalization(),
			layers.MaxPooling1D(2),
			layers.Dropout(0.3),
			
			layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
			layers.BatchNormalization(),
			layers.GlobalAveragePooling1D(),
			layers.Dropout(0.4),
			
			# Classification head
			layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
			layers.Dropout(0.2),
			layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
			layers.Dropout(0.2),
			layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
			layers.Dropout(0.2),
			
			layers.Dense(1, activation='sigmoid')
		])
		
		model.compile(
			optimizer=keras.optimizers.Adam(learning_rate=0.0005),
			loss=focal_loss(gamma=2.5, alpha=0.75),
			metrics=['accuracy',
					keras.metrics.Precision(name='precision'),
					keras.metrics.Recall(name='recall'),
					keras.metrics.AUC(name='auc')]
		)

		model.summary()
		print("\nUsing Focal Loss (gamma=2.5, alpha=0.75)")
		return model
	
	def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
		"""Train the model with AUC-centric callbacks."""
		print("\n" + "="*70)
		print("TRAINING")
		print("="*70)
		
		callbacks = [
			EarlyStopping(
				monitor='val_auc',
				patience=20,
				restore_best_weights=True,
				mode='max',
				verbose=1
			),
			ReduceLROnPlateau(
				monitor='val_auc',
				factor=0.5,
				patience=8,
				min_lr=1e-7,
				mode='max',
				verbose=1
			),
			ModelCheckpoint(
				'best_model_final.keras',
				monitor='val_auc',
				save_best_only=True,
				mode='max',
				verbose=1
			)
		]
		
		history = model.fit(
			X_train, y_train,
			validation_data=(X_val, y_val),
			epochs=epochs,
			batch_size=32,
			callbacks=callbacks,
			verbose=1
		)
		return history


def focal_loss(gamma=2.5, alpha=0.75):
    """Focal loss optimized for severe imbalance (binary)."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = alpha_factor * K.pow(1 - pt, gamma)
        bce = -K.log(pt)
        return K.mean(focal_weight * bce)
    return focal_loss_fixed		
