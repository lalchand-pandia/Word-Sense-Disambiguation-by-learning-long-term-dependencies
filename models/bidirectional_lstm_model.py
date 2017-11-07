
import tensorflow as tf
#from tensorflow.python.ops import rnn, rnn_cell, seq2seq
import numpy as np
import util.vocabmapping
import pickle
import math
from glove import *
word = "serve"
number_of_classes = 4
with open('util/vocab_'+word+'_sentences.txt', 'rb') as handle:
	init_word_vecs=pickle.load(handle)

init_emb = fill_with_gloves(init_word_vecs, 100)
#embedding_matrix=np.load("/home/lalchand/word_sense_disambiguation/100features_1minwords_3context_copy.npy")
class SentimentModel(object):
	'''
	Sentiment Model
	params:
	vocab_size: size of vocabulary
	hidden_size: number of units in a hidden layer
	num_layers: number of hidden lstm layers
	max_gradient_norm: maximum size of gradient
	max_seq_length: the maximum length of the input sequence
	learning_rate: the learning rate to use in param adjustment
	lr_decay:rate at which to decayse learning rate
	forward_only: whether to run backward pass or not
	'''
	def __init__(self, vocab_size, hidden_size, dropout,
	num_layers, max_gradient_norm, max_seq_length,
	learning_rate, lr_decay,batch_size, forward_only=False):
		self.num_classes = number_of_classes = 4
		self.vocab_size = vocab_size
		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
		self.learning_rate * lr_decay)
		initializer = tf.random_uniform_initializer(-1,1)
		self.batch_pointer = 0
		self.seq_input = []
		self.batch_size = batch_size
		self.seq_lengths = []
		self.projection_dim = hidden_size
		self.dropout = dropout
		self.max_gradient_norm = max_gradient_norm
		self.global_step = tf.Variable(0, trainable=False)
		self.max_seq_length = max_seq_length
	
		#seq_input: list of tensors, each tensor is size max_seq_length
		#target: a list of values betweeen 0 and 1 indicating target scores
		#seq_lengths:the early stop lengths of each input tensor
		self.str_summary_type = tf.placeholder(tf.string,name="str_summary_type")
		self.seq_input = tf.placeholder(tf.int32, shape=[None, max_seq_length],
		name="input")
		#print('in init ',max_seq_length)
		self.target = tf.placeholder(tf.float32, name="target", shape=[None,self.num_classes+1])
		self.seq_lengths = tf.placeholder(tf.int32, shape=[None],
		name="early_stop")

		self.dropout_keep_prob_embedding = tf.constant(self.dropout)
		self.dropout_keep_prob_lstm_input = tf.constant(self.dropout)
		self.dropout_keep_prob_lstm_output = tf.constant(self.dropout)
		'''def embedding_initializer(vec, dtype,partition_info=None):
			return tf.random_uniform([vocab_size, hidden_size],-.1, .1, dtype)'''

		'''with tf.variable_scope("embedding"):
		 	W = tf.get_variable(
		 		"W",
		 		[self.vocab_size, hidden_size],
				initializer=embedding_initializer,trainable=True)'''
		
		def embedding_initializer(vec, dtype,partition_info=None):
			return init_emb if init_emb is not None else tf.random_uniform([vocab_size, 100],-.1, .1, dtype)

		with tf.variable_scope("embedding"):
			W = tf.get_variable(
				"W",
				[self.vocab_size, 100],
				initializer=embedding_initializer)
			#embedded_tokens contains array of indices present in  self.seq_input and each array is of size vocab_size X 100 
			embedded_tokens = tf.nn.embedding_lookup(W, self.seq_input)
			
			embedded_tokens_drop = tf.nn.dropout(embedded_tokens, self.dropout_keep_prob_embedding)
			

			embedding_weights=tf.summary.histogram('embedding_weight_update ', W)

		rnn_input = [embedded_tokens_drop[:, i, :] for i in range(self.max_seq_length)]

		'''def lstm_cell():
      			return tf.contrib.rnn.BasicLSTMCell(
          			hidden_size, forget_bias=0.0, state_is_tuple=True)
		attn_cell = lstm_cell'''
    	#if is_training and config.keep_prob < 1:
		'''def attn_cell():
			return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.dropout_keep_prob_lstm_output)
		
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)], state_is_tuple=True)

		initial_state = cell.zero_state(self.batch_size, tf.float32)'''

		with tf.variable_scope("lstm") as scope:
			forward_cell = tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.LSTMCell(hidden_size,
								  initializer=tf.random_uniform_initializer(-1.0, 1.0),state_is_tuple=True),
								  input_keep_prob=self.dropout_keep_prob_lstm_input,
								  output_keep_prob=self.dropout_keep_prob_lstm_output)
			
			backward_cell = tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.LSTMCell(hidden_size,
								  initializer=tf.random_uniform_initializer(-1.0, 1.0),state_is_tuple=True),
								  input_keep_prob=self.dropout_keep_prob_lstm_input,
								  output_keep_prob=self.dropout_keep_prob_lstm_output)
			
	
			forward_cell = tf.contrib.rnn.MultiRNNCell([forward_cell for _ in range(num_layers)],state_is_tuple=True)
			backward_cell = tf.contrib.rnn.MultiRNNCell([backward_cell for _ in range(num_layers)],state_is_tuple=True)

			
			

			initial_forward_state = forward_cell.zero_state(self.batch_size, tf.float32)
			initial_backward_state = backward_cell.zero_state(self.batch_size, tf.float32)
			
			

			#rnn_output, rnn_state = tf.contrib.rnn.static_rnn(cell, rnn_input,initial_state=initial_state,sequence_length=self.seq_lengths)
			rnn_output, rnn_state_fw,rnn_state_bw = tf.contrib.rnn.static_bidirectional_rnn(forward_cell,backward_cell, rnn_input,initial_state_fw=initial_forward_state,initial_state_bw=initial_backward_state,sequence_length=self.seq_lengths)
			#rnn_concatenated_state=tf.concat(values, axis)
			forward_cell_weight=tf.summary.histogram(' forward _cell_weight ', rnn_state_fw)
			forward_mean = tf.reduce_mean(rnn_state_fw)

			backward_cell_weight=tf.summary.histogram(' backward_cell_weight ', rnn_state_bw)
			backward_mean = tf.reduce_mean(rnn_state_bw)
			forward_mean=tf.summary.scalar('forward_mean', forward_mean)
			backward_mean=tf.summary.scalar('backward_mean', backward_mean)
			

			rnn_concatenated_state = tf.concat([rnn_state_fw, rnn_state_bw],3)

			# states_list_fw = []
			# for state in rnn_state_fw[-1]:
			# 	states_list_fw.append(state)
			# avg_states_fw = tf.reduce_mean(tf.stack(states_list_fw), 0)
			# states_list_bw = []
			# for state in rnn_state_bw[-1]:
			# 	states_list_bw.append(state)
			# avg_states_bw = tf.reduce_mean(tf.stack(states_list_bw), 0)

		#****************************extra addition of hidden layer after blstm layer*************************************************
		#print('rnn_concatenated_state ',rnn_concatenated_state)
		state_size=50
		keep_prob=0.5
		'''with tf.variable_scope('hidden_layer', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
			hidden = tf.contrib.layers.fully_connected(rnn_concatenated_state, num_outputs=state_size, trainable=True,activation_fn=None)
			#if is_training:
			hidden = tf.nn.dropout(hidden, keep_prob)'''
		#print('hidden ',hidden)
		with tf.variable_scope("output_projection"):
			W = tf.get_variable(
				"W",
				[2*hidden_size, self.num_classes+1],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable(
				"b",
				[self.num_classes+1],initializer=tf.constant_initializer(0.1))

			self.scores = tf.nn.xw_plus_b(rnn_concatenated_state[-1][0], W, b)
			self.y = tf.nn.softmax(self.scores)
			self.predictions = tf.argmax(self.scores, 1)
			self.modified_targets=tf.argmax(self.target, 1)
			output_weight=tf.summary.histogram('output_projection_weights ', W)

		'''with tf.variable_scope("output_projection"):
			W = tf.get_variable(
				"W",
				[state_size, self.num_classes],
				initializer=tf.truncated_normal_initializer(stddev=0.1))
			b = tf.get_variable(
				"b",
				[self.num_classes],
				initializer=tf.constant_initializer(0.1))'''
			

			#self.scores = tf.nn.xw_plus_b(hidden[-1][0], W, b)
			#print(self.scores,' self.scores ')
			
			#print(' self.predictions ',self.predictions)

		with tf.variable_scope("loss"):
			self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.target, name="ce_losses")
			self.total_loss = tf.reduce_sum(self.losses)
			self.mean_loss = tf.reduce_mean(self.losses)

		with tf.variable_scope("accuracy"):
			
			self.correct_predictions = tf.equal(self.predictions, tf.argmax(self.target, 1))
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

		params = tf.trainable_variables()
		if not forward_only:
			with tf.name_scope("train") as scope:
				opt = tf.train.AdamOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
			with tf.name_scope("grad_norms") as scope:
				grad_summ = tf.summary.scalar("grad_norms", norm)
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
			loss_summ = tf.summary.scalar("{0}_loss".format(self.str_summary_type), self.mean_loss)
			acc_summ = tf.summary.scalar("{0}_accuracy".format(self.str_summary_type), self.accuracy)
			#new addition to view weights

			self.merged = tf.summary.merge([loss_summ, acc_summ,output_weight,embedding_weights,forward_cell_weight,backward_cell_weight,forward_mean,backward_mean])
		self.saver = tf.train.Saver()
		tf.global_variables_initializer().run()


	def getBatch(self, test_data=False):
		'''
		Get a random batch of data to preprocess for a step
		not sure how efficient this is...

		Input:
		data: shuffled batchxnxm numpy array of data
		train_data: flag indicating whether or not to increment batch pointer, in other
			word whether to return the next training batch, or cross val data

		Returns:
		A numpy arrays for inputs, target, and seq_lengths

		'''
		#batch_inputs = []
		if not test_data:
			batch_inputs = self.train_data[self.train_batch_pointer]#.transpose()
			#for i in range(self.max_seq_length):
			#	batch_inputs.append(temp[i])
			targets = self.train_targets[self.train_batch_pointer]
			seq_lengths = self.train_sequence_lengths[self.train_batch_pointer]
			self.train_batch_pointer += 1
			self.train_batch_pointer = self.train_batch_pointer % len(self.train_data)
			return batch_inputs, targets, seq_lengths
		else:
			batch_inputs = self.test_data[self.test_batch_pointer]#.transpose()
			#for i in range(self.max_seq_length):
			#	batch_inputs.append(temp[i])
			targets = self.test_targets[self.test_batch_pointer]
			seq_lengths = self.test_sequence_lengths[self.test_batch_pointer]
			self.test_batch_pointer += 1
			self.test_batch_pointer = self.test_batch_pointer % len(self.test_data)
			return batch_inputs, targets, seq_lengths

	def initData(self, data, train_start_end_index, test_start_end_index):
		'''
		Split data into train/test sets and load into memory
		'''
		self.train_batch_pointer = 0
		self.test_batch_pointer = 0
		#cutoff non even number of batches
		#print('data ',data.shape)
		targets = (data.transpose()[-2]).transpose()
		#print('targets ',targets)
		#print(np.unique(targets))
		#+1 for unknown class to remedy padding
		onehot = np.zeros((len(targets), self.num_classes+1))
		#print('onehot ',onehot)
		#print len(targets)

		onehot[np.arange(len(targets)), targets] = 1
		#print('onehot ',onehot)
		sequence_lengths = (data.transpose()[-1]).transpose()
		#print(data.shape)
		data = (data.transpose()[0:-2]).transpose()
		#print(data.shape)

		self.train_data = data[train_start_end_index[0]: train_start_end_index[1]+1]
		self.test_data = data[test_start_end_index[0]:test_start_end_index[1]+1]
		#pad test_data
		pad_amount=int(math.ceil(len(self.test_data)/float(self.batch_size)))*self.batch_size-len(self.test_data)
		#print('pad_maount ',pad_amount)
		auxiliary_test_data=np.zeros([pad_amount,100+2],dtype=np.int32)
		auxiliary_test_data.fill(util.vocabmapping.VocabMapping(word).getIndex("<PAD>"))
		auxiliary_test_data[:,100]=self.num_classes
		auxiliary_test_data[:,101]=100
		test_data=(auxiliary_test_data.transpose()[0:-2]).transpose()
		test_data_sequence_length=(auxiliary_test_data.transpose()[-1]).transpose()
		test_data_target=(auxiliary_test_data.transpose()[-2]).transpose()
		targets=np.append(targets,test_data_target,axis=0)
		#print('targets ',targets,' len(targets) ',len(targets))
		onehot=np.zeros((len(targets),self.num_classes+1))
		onehot[np.arange(len(targets)),targets]=1
		#print('auxiliary_data shape ',test_data.shape)
		#print('test_Data shape ',self.test_data.shape)
		#print('auxiliary_data ',auxiliary_test_data)
		data_len=len(data)
		#test_n_batches_float = data_len / float(self.batch_size)
		#test_n_batches = int(math.ceil(test_n_batches_float))
		#self.test_num_batch = len(self.test_data) / test_n_batches
		self.test_data=np.append(self.test_data, test_data,axis=0)
		#self.test_num_batch = len(self.test_data) / self.batch_size

		num_train_batches = len(self.train_data) / self.batch_size
		#num_test_batches = len(self.test_data) / self.batch_size
		num_test_batches = len(self.test_data) / self.batch_size
		#print('num_test_batches ',num_test_batches)
		#num_test_batches = len(self.test_data)
		train_cutoff = len(self.train_data) - (len(self.train_data) % self.batch_size)
		#test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
		#test_cutoff = len(self.test_data) - (len(self.test_data) % self.batch_size)
		#print(train_cutoff,' test_cutoff ',test_cutoff)
		self.train_data = self.train_data[:train_cutoff]
		#self.test_data = self.test_data[:test_cutoff]
		#print(self.train_data.shape)
		self.train_sequence_lengths = sequence_lengths[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		self.train_sequence_lengths = np.split(self.train_sequence_lengths, num_train_batches)
		self.train_targets = onehot[train_start_end_index[0]:train_start_end_index[1]][:train_cutoff]
		#print(' self.train_targets ',self.train_targets)
		#print(self.train_targets.shape)
		#split the target array into subarrays of num_train_batches
		self.train_targets = np.split(self.train_targets, num_train_batches)
		#split the data array into subarrays of num_train_batches
		self.train_data = np.split(self.train_data, num_train_batches)

		print "Test size is: {0}, splitting into {1} batches".format(len(self.test_data), num_test_batches)
		self.test_data = np.split(self.test_data, num_test_batches)
		#self.test_targets = onehot[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
		#print('one hot ',onehot.shape,' test_start_end_index[0] ',test_start_end_index[0],' onehot.shape[0] ',onehot.shape[0])

		self.test_targets = onehot[test_start_end_index[0]:onehot.shape[0]]

		#print('self.test_targets ',self.test_targets.shape)
		self.test_targets = np.split(self.test_targets, num_test_batches)
		#self.test_sequence_lengths = sequence_lengths[test_start_end_index[0]:test_start_end_index[1]][:test_cutoff]
		self.test_sequence_lengths = sequence_lengths[test_start_end_index[0]:test_start_end_index[1]+1]
		self.test_sequence_lengths=np.append(self.test_sequence_lengths,test_data_sequence_length)
		#print('self.test_sequence_lengths ',self.test_sequence_lengths.shape)
		self.test_sequence_lengths = np.split(self.test_sequence_lengths, num_test_batches)

	def step(self, session, inputs, targets, seq_lengths, forward_only=False):
		'''
		Inputs:
		session: tensorflow session
		inputs: list of list of ints representing tokens in review of batch_size
		output: list of sentiment scores
		seq_lengths: list of sequence lengths, provided at runtime to prevent need for padding

		Returns:
		merged_tb_vars, loss, none
		or (in forward only):
		merged_tb_vars, loss, outputs
		'''
		#print(' targets  ',targets)
		input_feed = {}
		#for i in xrange(self.max_seq_length):
		input_feed[self.seq_input.name] = inputs

		#print inputs.shape
		#print('targets ',targets)
		#print('target name ',self.target.name)
		input_feed[self.target.name] = targets
		#print('input_feed[target]',input_feed[self.target.name])
		#print input_feed[self.target.name].shape
		#print(targets,' in step ')
		input_feed[self.seq_lengths.name] = seq_lengths
		#print input_feed[self.seq_lengths.name]
		if not forward_only:
			input_feed[self.str_summary_type.name] = "train"
			output_feed = [self.merged, self.mean_loss, self.update]
		else:
			input_feed[self.str_summary_type.name] = "test"
			output_feed = [self.merged, self.mean_loss, self.y, self.accuracy,self.predictions,self.modified_targets]
		outputs = session.run(output_feed, input_feed)

		if not forward_only:
			return outputs[0], outputs[1], None
		else:
			return outputs[0], outputs[1], outputs[2], outputs[3],outputs[4],outputs[5]
