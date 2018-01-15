import nltk
import pickle
import sys
word = sys.argv[1]
number_of_senses = int(sys.argv[2])
def createVocab(max_vocab_size):
    print "Creating vocab mapping..."
    dic = {}
    f=open("wsd_datasets/"+word+"/"+word+"_sentences.txt","r")
    f1=open("wsd_datasets/"+word+"/"+word+"_senses.txt","r")
    for line in f.readlines():
        

        tokens = line.lower().decode('utf-8').split()
        for t in tokens:
            if t not in dic:
                        dic[t] = 1
            else:
                        dic[t] += 1
    d = {}
    d_index_2_word = {}
    counter = 0
    for w in sorted(dic, key=dic.get, reverse=True):
        d[w] = counter
        d_index_2_word[counter] = w
        counter += 1
        #take most frequent 50k tokens
        if counter >=max_vocab_size:
            break
    #add out of vocab token and pad token
    d["<UNK>"] = counter
    d_index_2_word[counter] = "<UNK>"
    counter +=1
    d["<PAD>"] = counter
    d_index_2_word[counter] = "<PAD>"

    with open('util/vocab_'+word+'_sentences.txt', 'w') as handle:
        pickle.dump(d, handle)
    with open("util/"+word+"_index_2_word_map.txt",'w') as handle2:
        pickle.dump(d_index_2_word, handle2)



    dic1={}
    for line in f1.readlines():
        

        tokens =line.lower().decode('utf-8').split()
        for t in tokens:
            if t not in dic1:
                        dic1[t] = 1
            else:
                        dic1[t] += 1
    d1 = {}
    d1_index_2_word = {}
    counter = 0
    for w in sorted(dic1, key = dic1.get, reverse=True):
        d1[w] = counter
        d1_index_2_word[counter] = w
        counter += 1
        #take most frequent 50k tokens
        if counter >= max_vocab_size:
            break
    #add out of vocab token and pad token
    
    
    d1['unknown_class'] = number_of_senses
    d1_index_2_word[number_of_senses] = 'unknown_class'
    for key,value in d1.items():
            print key,' ',value

    with open('util/vocab_'+word+'_senses.txt', 'w') as handle1:
        pickle.dump(d1, handle1)
    with open("util/"+word+"_index_2_word_senses_map.txt",'w') as handle3:
        pickle.dump(d1_index_2_word, handle3)


createVocab(20000)
