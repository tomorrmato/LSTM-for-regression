import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 50
n_steps = 15
input_size, output_size = 1, 1
cell_size = 10
learning_rate = 0.005
batch_start = 0

# get data batch by batch
def get_batch():
	global batch_start, n_steps
	xs = np.arange(batch_start, batch_start+n_steps*batch_size).reshape((batch_size, n_steps)) / (10*np.pi)
	seq = np.sin(xs)
	res = np.cos(xs)
	batch_start += n_steps
	return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

class LSTM(object):
	def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
		self.n_steps = n_steps
		self.input_size = input_size
		self.output_size = output_size
		self.cell_size = cell_size
		self.batch_size = batch_size
		self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size])
		self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size])
		self.add_input_layer()
		self.add_cell()
		self.add_output_layer()
		self.compute_loss()
		self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	#initialize random variables for weight
	def generate_weight(self, shape):
		return tf.Variable(tf.random_normal(shape=shape, mean=0, stddev=1))

	#initialize random variables for bias
	def generate_bias(self, shape):
		return tf.Variable(tf.random_normal(shape=shape, mean=0, stddev=1))

	#construct input layer
	def add_input_layer(self):
		x_in = tf.reshape(self.xs, [-1, self.input_size])
		w_in = self.generate_weight([self.input_size, self.cell_size])
		b_in = self.generate_bias([self.cell_size, ])
		y_in = tf.matmul(x_in, w_in)+b_in
		self.y_in = tf.reshape(y_in, [-1, self.n_steps, self.cell_size])
		print "input layer:", "x_in:", x_in.shape, "y_in:",self.y_in.shape

	#construct LSTM cell, initialize RNN state
	def add_cell(self):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
		self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
		self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
			lstm_cell, self.y_in, initial_state=self.cell_init_state, time_major=False)

	#construct output layer
	def add_output_layer(self):
		x_out = tf.reshape(self.cell_outputs, [-1, self.cell_size])
		w_out = self.generate_weight([self.cell_size, self.output_size])
		b_out = self.generate_bias([self.output_size, ])
		self.ys_pred = tf.matmul(x_out, w_out)+b_out
		print "output layer:", "x_out:", x_out.shape, "ys_pred", self.ys_pred.shape

	#calculate loss function 
	def compute_loss(self):	
		print 'ys:', self.ys.shape, 'ys_pred:', self.ys_pred.shape
		cost = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[tf.reshape(self.ys_pred, [-1])], 
			[tf.reshape(self.ys, [-1])],
			[tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
			average_across_timesteps=True,
			softmax_loss_function=self.ms_error)
		self.loss = tf.reduce_sum(cost)
	
	#softmax_loss_function in sequence_loss_by_example
	def ms_error(self, ys, ys_pred):
		return tf.square(tf.subtract(ys, ys_pred))

if __name__ == '__main__':
	model = LSTM(n_steps, input_size, output_size, cell_size, batch_size)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	plt.ion()
	plt.show()
	for i in range(200):
		seq, res, xs = get_batch()
		if i == 0:		
			feed_dict = {model.xs: seq, model.ys: res}
		else:
			feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}

		_, cost, state, pred = sess.run(
			[model.train_op, model.loss, model.cell_final_state, model.ys_pred],
			feed_dict=feed_dict)

		# plotting
		plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:n_steps], 'b--')
		plt.ylim((-1.2, 1.2))
		plt.draw()
		plt.pause(0.3)

		if i % 20 == 0:
			print('cost: ', round(cost, 4))
	plt.savefig('plot figures')
