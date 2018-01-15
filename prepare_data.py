import util.vocabmapping
import pickle
import numpy as np
import nltk
import sys
word = sys.argv[1]

def createProcessedDataFile(vocab_mapping, max_seq_length):
    count = 0
    data = np.array([i for i in range(max_seq_length + 2)])
    f=open("wsd_datasets/"+word+"/"+word+"_sentences.txt","r")
    f1=open("wsd_datasets/"+word+"/"+word+"_senses.txt","r")
    
    target=[]
    for line in f1.readlines():
    	line=line.strip().lower()
    	if line:
    		tokens =line.lower().decode('utf-8').split()
    		indices = [vocab_mapping.getIndex_target(j) for j in tokens]
    		target.append(indices[0])

  
    k = 0
    
    for line in f.readlines():
    	tokens =line.lower().decode('utf-8').split()
    	numTokens = len(tokens)
    	indices = [vocab_mapping.getIndex(j) for j in tokens]
    	if len(indices) < max_seq_length:
        	indices = indices + [vocab_mapping.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
        else:
        	indices = indices[0:max_seq_length]
     
        indices.append(target[k])
        k+=1
        
        indices.append(min(numTokens, max_seq_length))
     
        data = np.vstack((data, indices))
        indices = []
        
 
    data = data[1::]
    
    
    np.save(word+"_with_glove_vectors_100.npy", data)
    


vocab = util.vocabmapping.VocabMapping(word)
createProcessedDataFile(vocab,100)
