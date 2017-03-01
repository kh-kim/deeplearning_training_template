import json
import numpy as np
import h5py

from os import path

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

class LearningSuite(object):

	def __init__(self, json_filename):
		f = open(json_filename, 'r')
		js = json.loads(f.read())
		f.close()

		self.MAX_EPOCH = int(js.get("MAX_EPOCH")) if js.get("MAX_EPOCH") != None else 100
		self.SAMPLES_PER_EPOCH = int(js.get("SAMPLES_PER_EPOCH")) if js.get("SAMPLES_PER_EPOCH") != None else -1
		self.SAMPLES_PER_TEST = int(js.get("SAMPLES_PER_TEST")) if js.get("SAMPLES_PER_TEST") != None else -1
		self.BATCH_SIZE = int(js.get("BATCH_SIZE")) if js.get("BATCH_SIZE") != None else 128
		self.OPTIMIZER = js.get("OPTIMIZER") if js.get("OPTIMIZER") != None else "sgd"
		self.LOSS = js.get("LOSS") if js.get("LOSS") != None else "mse"
		self.LEARNING_RATE = float(js.get("LEARNING_RATE")) if js.get("LEARNING_RATE") != None else 0.025

	def train(self, model_builder, train_iterator, test_iterator, verbose = 1, model_check_point = None, early_stopping = None):
		self.model = model_builder.get_model()
		self.train_iterator = train_iterator
		self.test_iterator = test_iterator

		callbacks = []
		if model_check_point != None:
			callbacks.append(model_check_point)
		if early_stopping != None:
			callbacks.append(early_stopping)

		print "MAX EPOCH: %d" % self.MAX_EPOCH
		print "SAMPLES PER EPOCH: %d" % self.SAMPLES_PER_EPOCH
		print "SAMPLES PER TEST: %d" % self.SAMPLES_PER_TEST
		print "BATCH_SIZE: %d" % self.BATCH_SIZE
		print "LOSS: %s" % self.LOSS
		print "OPTIMIZER: %s" % self.OPTIMIZER

		if self.OPTIMIZER == "sgd":
			sgd = SGD(lr = self.LEARNING_RATE, decay = 1e-6, momentum = 0.9, nesterov = True)
			self.OPTIMIZER = sgd

			print "LEARNING RATE: %f" % self.LEARNING_RATE

		self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=['accuracy'])

		history = self.model.fit_generator(train_iterator,
				samples_per_epoch = self.SAMPLES_PER_EPOCH,
				nb_epoch = self.MAX_EPOCH,
				validation_data = test_iterator,
				nb_val_samples = self.SAMPLES_PER_TEST,
				verbose = verbose,
				callbacks = callbacks
				)

		return history

	def predict_next(self, iterator, model_builder = None):
		if self.model == None and model_builder != None:
			self.model = model_builder.get_model()
			self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=['accuracy'])

		x = iterator.next()[0]
		y = self.model.predict(x)

		return x, y

	def get_model(self):
		if self.model != None:
			return self.model
		else:
			return None

class AbstractModelBuilder(object):

	def __init__(self, json_filename = None, weights_path = None):
		self.weights_path = weights_path

		if json_filename != None:
			f = open(json_filename, 'r')
			js = json.loads(f.read())
			f.close()

			self.INPUT = js.get('INPUT') if js.get('INPUT') != None else None
			self.OUTPUT = js.get('OUTPUT') if js.get('OUTPUT') != None else None

	def get_model(self):
		model = self.define_model()

		if self.weights_path and path.isfile(self.weights_path):
			try:
				model.load_weights(self.weights_path)
			except Exception, e:
				print e

		return model

	def define_model(self):
		raise NotImplementedError("You need to define your own model architecture.")

