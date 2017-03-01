import numpy as np

from keras.preprocessing.image import Iterator

from template_trainer import LearningSuite, AbstractModelBuilder

class SampleModelBuilder(AbstractModelBuilder):

	def define_model(self):
		from keras.models import Model
		from keras.layers import Input, Dense

		s = Input(shape = (int(self.INPUT), ))
		l = Dense(32, activation = 'relu')(s)
		l = Dense(32, activation = 'relu')(l)
		v = Dense(int(self.OUTPUT), activation = 'softmax')(l)

		model = Model(input = s, output = v)

		return model


class SampleIterator(Iterator):

	def __init__(self, batch_size, sample_size, shuffle = True, seed = None):
		from random import random

		data = []
		for i in xrange(sample_size):
			x_ = int(random() * 1000)

			x = self.convert_2_binary(x_)
			y = [0] * 4

			if x_ % 3 == 0 and x_ % 5 == 0:
				y[3] = 1
			elif x_ % 5 == 0:
				y[2] = 1
			elif x_ % 3 == 0:
				y[1] = 1
			else:
				y[0] = 1

			data.append((np.array(x), np.array(y)))

		self.data = data
		N = len(data)

		super(SampleIterator, self).__init__(N, batch_size, shuffle, seed)
		

	def next(self):
		with self.lock:
			index_array, current_index, current_batch_size = next(self.index_generator)

		batch_x = []
		batch_y = []

		for i, j in enumerate(index_array):
			x, y = self.data[j]
			batch_x.append(x)
			batch_y.append(y)

		return np.array(batch_x), np.array(batch_y)


	def convert_2_binary(self, dec):
		x = [0] * 10

		t = dec
		cnt = 1
		while True:
			r = t % 2
			t = t / 2

			x[-cnt] = r
			cnt += 1

			if t < 2:
				x[-cnt] = t
				break

		return x

if __name__ == '__main__':
	suite = LearningSuite('./hparam.json')
	model_builder = SampleModelBuilder('./hparam.json')
	train_iterator = SampleIterator(suite.BATCH_SIZE, 900)
	test_iterator = SampleIterator(suite.BATCH_SIZE, 100)

	suite.train(model_builder, train_iterator, test_iterator, verbose = 2)

	test_iterator = SampleIterator(suite.BATCH_SIZE, 100, shuffle = False)
	print suite.predict_next(test_iterator)
