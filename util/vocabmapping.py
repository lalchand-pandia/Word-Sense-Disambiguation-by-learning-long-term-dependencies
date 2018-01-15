import pickle

class VocabMapping:
    def __init__(self, word):
        
        with open("util/vocab_"+word+"_sentences.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())
        with open("util/vocab_"+word+"_senses.txt","rb") as handle1:
            self.dic1 = pickle.loads(handle1.read())

    def getIndex(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]
    def getIndex_target(self, token):
        try:
            return self.dic1[token]
        except:
            return self.dic1["<UNK>"]

    def getSize(self):
        return len(self.dic)
    def getSize_target(self):
        return len(self.dic1)
