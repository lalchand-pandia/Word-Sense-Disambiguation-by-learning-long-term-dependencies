'''
I used mainly the tensorflow translation example:
https://github.com/tensorflow/tensorflow/

and semi-based this off the sentiment analyzer here:
http://deeplearning.net/tutorial/lstm.html

Written by: Dominik Kaukinen
'''
import tensorflow as tf
import pickle
from tensorflow.python.platform import gfile
from glove import *
import numpy as np
import sys
import math
import os
import ConfigParser
import random
import time
from six.moves import xrange
#import util.dataprocessor
import util.hyperparams as hyperparams
#import models.sentiment
import models.bidirectional_lstm_model
import util.vocabmapping
import time
import operator
#from data.senseeval.first.vocabmapping import VocabMapping
#Defaults for network 
import sys
word = sys.argv[1]

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("config_file", "config.ini", "Path to configuration file with hyper-parameters.")
flags.DEFINE_string("data_dir", "data/", "Path to main data directory.")
flags.DEFINE_string("checkpoint_dir", "data/checkpoints/", "Directory to store/restore checkpoints")



def main():
	hyper_params = check_get_hyper_param_dic()
	#creates numpy list of word-indices and store in files
	#util.dataprocessor.run(hyper_params["max_seq_length"],
	#	hyper_params["max_vocab_size"])

	#create model
	print "Creating model with..."
	print "Number of hidden layers: {0}".format(hyper_params["num_layers"])
	print "Number of units per layer: {0}".format(hyper_params["hidden_size"])
	print "Dropout: {0}".format(hyper_params["dropout"])
	vocabmapping = util.vocabmapping.VocabMapping(word)
	#vocabmapping=VocabMapping()
	vocab_size = vocabmapping.getSize()
	print "Vocab size is: {0}".format(vocab_size)
	path = os.path.join(FLAGS.data_dir, "processed/")
	
	input_File=word+"_with_glove_vectors_100.npy"
	
	data1 = np.load(input_File)
	average_accuracy=0.0
	s=time.time()
	for  i in xrange(0,1):
		np.random.shuffle(data1)
	#data = data[:3000]
		num_batches = len(data1) / hyper_params["batch_size"]
	# 70/30 splir for train/test
		train_start_end_index = [0, int(hyper_params["train_frac"] * len(data1))]
		test_start_end_index = [int(hyper_params["train_frac"] * len(data1)) + 1, len(data1) - 1]
		print('train_start_end_index ',train_start_end_index,' test_start_end_index ',test_start_end_index)
		print "Number of training examples per batch: {0}, \
		\nNumber of batches per epoch: {1}".format(hyper_params["batch_size"],num_batches)
	
	
		with tf.Session() as sess:
			
			#writer.add_graph(sess.graph)
			model = create_model(sess, hyper_params, vocab_size)
			#initialized filewriter
			writer = tf.summary.FileWriter("/tmp/tb_logs", sess.graph)
			tf.get_variable_scope().reuse_variables()
	#train model and save to checkpoint
			print "Beggining training..."
			print "Maximum number of epochs to train for: {0}".format(hyper_params["max_epoch"])
			print "Batch size: {0}".format(hyper_params["batch_size"])
			print "Starting learning rate: {0}".format(hyper_params["learning_rate"])
			print "Learning rate decay factor: {0}".format(hyper_params["lr_decay_factor"])

			step_time, loss = 0.0, 0.0
			previous_losses = []
			tot_steps = num_batches * hyper_params["max_epoch"]
			model.initData(data1, train_start_end_index, test_start_end_index)
			X=[]
			Y=[]
			#starting at step 1 to prevent test set from running after first batch
			prediction_result=[]
			target_result=[]
			input_sentences_data=[]
			for step in xrange(1, tot_steps):
			# Get a batch and make a step.
				start_time = time.time()
			#target contain a batch of size 64*num_classes
				inputs, targets, seq_lengths = model.getBatch()
			#print(targets,' in train ')
				str_summary, step_loss, _ = model.step(sess, inputs, targets, seq_lengths, False)
				#writer.add_summary(str_summary, step)

				step_time += (time.time() - start_time) / hyper_params["steps_per_checkpoint"]
				loss += step_loss / hyper_params["steps_per_checkpoint"]

			# Once in a while, we save checkpoint, print statistics, and run evals.
				if step % hyper_params["steps_per_checkpoint"] == 0:
					writer.add_summary(str_summary, step)
				# Print statistics for the previous epoch.
					print ("global step %d learning rate %.7f step-time %.2f loss %.4f"
					% (model.global_step.eval(), model.learning_rate.eval(),
					step_time, loss))
				# Decrease learning rate if no improvement was seen over last 3 times.
					if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
						sess.run(model.learning_rate_decay_op)
					previous_losses.append(loss)
					# Save checkpoint and zero timer and loss.
					step_time, loss, test_accuracy = 0.0, 0.0, 0.0
				# Run evals on test set and print their accuracy.
					print "Running test set"
				#X=[]
				#Y=[]
				#prediction_result=[]
				#target
					prediction_in_test=[]
					target_in_test=[]
					input_in_test=[]
					for test_step in xrange(len(model.test_data)):
						inputs, targets, seq_lengths = model.getBatch(True)
					#print(targets,' in test ')
						str_summary, test_loss, _, accuracy,predictions,modified_target= model.step(sess, inputs, targets, seq_lengths, True)
						loss += test_loss
						test_accuracy += accuracy
						prediction_in_test.append(predictions)
						target_in_test.append(modified_target)
						input_in_test.append(inputs)
					#print(' scores ',scores,' y ',y)
					'''if step==tot_steps-1:
						X.append(prediction_result)
						Y.append(target_result)'''
					prediction_result.append(prediction_in_test)
					target_result.append(target_in_test)
					input_sentences_data.append(input_in_test)
				#print('targets ',targets)
				#print('predictions ',predictions)
					normalized_test_loss, normalized_test_accuracy = loss / len(model.test_data), test_accuracy / len(model.test_data)
					checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "word_sense_disambiguation{0}.ckpt".format(normalized_test_accuracy))
					model.saver.save(sess, checkpoint_path, global_step=model.global_step)
					writer.add_summary(str_summary, step)
					print "Avg Test Loss: {0}, Avg Test Accuracy: {1}".format(normalized_test_loss, normalized_test_accuracy)
					print "-------Step {0}/{1}------".format(step,tot_steps)
					loss = 0.0
					if test_step==len(model.test_data)-1:
						average_accuracy+=normalized_test_accuracy
					sys.stdout.flush()
		#X=np.asarray(X)
		#Y=np.asarray(Y)
		#np.save("out.npy",X)
		#print(len(prediction_result))
		#print(len(target_result))
		#print(prediction_result)
		#print(target_result)
			input_sentences_data=np.asarray(input_sentences_data)
			a=np.asarray(prediction_result)
			b=np.asarray(target_result)
			np.save("out_predicted.npy", a)
			np.save("result_predicted.npy", b)
			np.save("out1.npy",Y)
			pred=[]
			target=[]
			input_sentences_data_list=[]
			for i in input_sentences_data[tot_steps/hyper_params["steps_per_checkpoint"]-2]:
				for j in i:
					input_sentences_data_list.append(j)
			for i in a[tot_steps/hyper_params["steps_per_checkpoint"]-2]:
				for j in i:
					pred.append(j)
			for i in b[tot_steps/hyper_params["steps_per_checkpoint"]-2]:
				for j in i:
					target.append(j)
			with open("util/"+word+"_index_2_word_map.txt", "rb") as handle:
				dic_sentences = pickle.loads(handle.read())
			with open("util/"+word+"_index_2_word_senses_map.txt","rb") as handle1:
				dic1_senses=pickle.loads(handle1.read())
			f2=open(word+'_mismatch_result.txt','w')
			for i ,j ,k in zip(input_sentences_data_list,pred,target):
				if j!=k:
				#print(i, j, k)
					sentence=[]
					for l in i:
						f2.write(str(dic_sentences[l])+' ')
						sentence.append(dic_sentences[l])

				#print(sentence)
				#print(' prediction ',dic1_senses[j])
					f2.write(' MODEL_PREDICTION '+str(dic1_senses[j])+' GROUND_TRUTH '+str(dic1_senses[k])+' ')
				#print(' target ',dic1_senses[k])
				f2.write('\n')

			from sklearn.metrics import confusion_matrix
			cnf_matrix = confusion_matrix(target,pred)
		
			import matplotlib.pyplot as plt
		
			import itertools
			plt.ylabel('True label')
			plt.xlabel('Predicted label')
			with open("util/vocab_"+word+"_sentences.txt", "rb") as handle:
				dic = pickle.loads(handle.read())
			classes = [i[0] for i in sorted(dic.items() ,key = operator.itemgetter(1),reverse =True)]
		#classes=["HARD1","HARD2","HARD3","cord","division","formation","interest_1","interest_2","interest_3","interest_4","interest_5","interest_6","phone","product","text"]
			#classes=["interest_6","interest_5","interest_1","interest_4","interest_3","interest_2"]
			#classes=["product","phone","text","division","cord","formation"]
			'''classes=["serve10","serve12","serve2","serve6","unknown_class"]
			classes=["volume%1:10:00::","volume%1:07:03::","unknown_class"]
			classes=["line%1:04:01::","line%1:06:07::","line%1:09:00::","line%1:15:02::","line%1:06:09::","line%1:06:03::","line%1:14:02::","line%1:04:00::","unknown_class"]
			classes=['direction%1:04:00::','direction%1:10:01::','direction%1:09:00::','unknown_class']
			classes=['paper%1:10:02::','paper%1:27:00::','paper%1:10:03::','paper%1:10:00::','unknown_class']
			classes=['front%1:26:00::','front%1:15:01::','front%1:15:03::','front%1:15:00::','unknown_class']
			classes=['follow%2:38:00::','follow%2:38:01::','follow%2:41:00::','follow%2:42:02::','follow%2:40:00::','unknown_class']
			classes=['consider%2:31:00::','consider%2:39:00::','consider%2:32:00::','consider%2:32:02::','unknown_class']'''
			
            
            
			#classes=["hard1","hard2","hard3","unknown_class"]
			cm=cnf_matrix
			cmap=plt.cm.Blues

			plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
			plt.title("Confusion matrix")
			plt.colorbar()
			tick_marks = np.arange(len(classes))
			plt.xticks(tick_marks, classes, rotation=45)
			plt.yticks(tick_marks, classes)

    
        

			print(cnf_matrix)

			thresh = cnf_matrix.max() / 2.
			for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
		        	plt.text(j, i, cnf_matrix[i, j],
		                 horizontalalignment="center",
		                 color="white" if cnf_matrix[i, j] > thresh else "black")

		#plt.figure()
			plt.savefig('wsd_datasets/'+word+'/'+word+'_confusion_matrix_bidirectional_lstm_glove.png')
			#plt.show()
	
	end=time.time()
	print('time in training ',end-s)

def create_model(session, hyper_params, vocab_size):
	#init_emb = fill_with_gloves(word_to_id, conf['embedding_size'])
	model = models.bidirectional_lstm_model.SentimentModel(vocab_size,
		hyper_params["hidden_size"],
		hyper_params["dropout"],
		hyper_params["num_layers"],
		hyper_params["grad_clip"],
		hyper_params["max_seq_length"],
		hyper_params["learning_rate"],
		hyper_params["lr_decay_factor"],
		hyper_params["batch_size"])
	'''model=models.sentiment.SentimentModel(vocab_size,
		hyper_params["hidden_size"],
		hyper_params["dropout"],
		hyper_params["num_layers"],
		hyper_params["grad_clip"],
		hyper_params["max_seq_length"],
		hyper_params["learning_rate"],
		hyper_params["lr_decay_factor"],
		hyper_params["batch_size"])'''
	'''ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
	if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from {0}".format(ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:'''
	print "Created model with fresh parameters."
	session.run(tf.global_variables_initializer())
	return model

def read_config_file():
	'''
	Reads in config file, returns dictionary of network params
	'''
	config = ConfigParser.ConfigParser()
	config.read(FLAGS.config_file)
	dic = {}
	wsd_section = "WSD_network_params"
	general_section = "general"
	dic["num_layers"] = config.getint(wsd_section, "num_layers")
	dic["hidden_size"] = config.getint(wsd_section, "hidden_size")
	dic["dropout"] = config.getfloat(wsd_section, "dropout")
	dic["batch_size"] = config.getint(wsd_section, "batch_size")
	dic["train_frac"] = config.getfloat(wsd_section, "train_frac")
	dic["learning_rate"] = config.getfloat(wsd_section, "learning_rate")
	dic["lr_decay_factor"] = config.getfloat(wsd_section, "lr_decay_factor")
	dic["grad_clip"] = config.getint(wsd_section, "grad_clip")
	dic["use_config_file_if_checkpoint_exists"] = config.getboolean(general_section,
		"use_config_file_if_checkpoint_exists")
	dic["max_epoch"] = config.getint(wsd_section, "max_epoch")
	dic ["max_vocab_size"] = config.getint(wsd_section, "max_vocab_size")
	dic["max_seq_length"] = config.getint(general_section,
		"max_seq_length")
	dic["steps_per_checkpoint"] = config.getint(general_section,
		"steps_per_checkpoint")
	return dic

def check_get_hyper_param_dic():
	'''
	Retrieves hyper parameter information from either config file or checkpoint
	'''
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
	hyper_params = read_config_file()
	if serializer.checkExists():
		if serializer.checkChanged(hyper_params):
			if not hyper_params["use_config_file_if_checkpoint_exists"]:
				hyper_params = serializer.getParams()
				print "Restoring hyper params from previous checkpoint..."
			else:
				new_checkpoint_dir = "{0}_hidden_size_{1}_numlayers_{2}_dropout_{3}".format(
				int(time.time()),
				hyper_params["hidden_size"],
				hyper_params["num_layers"],
				hyper_params["dropout"])
				new_checkpoint_dir = os.path.join(FLAGS.checkpoint_dir,
					new_checkpoint_dir)
				os.makedirs(new_checkpoint_dir)
				FLAGS.checkpoint_dir = new_checkpoint_dir
				serializer = hyperparams.HyperParameterHandler(FLAGS.checkpoint_dir)
				serializer.saveParams(hyper_params)
		else:
			print "No hyper parameter changed detected, using old checkpoint..."
	else:
		serializer.saveParams(hyper_params)
		print "No hyper params detected at checkpoint... reading config file"
	return hyper_params

if __name__ == '__main__':
	main()
