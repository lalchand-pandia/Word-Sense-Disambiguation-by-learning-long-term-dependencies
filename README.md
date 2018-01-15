# Word-Sense-Disambiguation-by-learning-long-term-dependencies
Word sense disambiguation by using recurrent networks like Bidirectional LSTM

# Requirements
 1. Tensorflow
 2. pickle

# Train the model

git clone https://github.com/lalchand-pandia/Word-Sense-Disambiguation-by-learning-long-term-dependencies.git

cd Word-Sense-Disambiguation-by-learning-long-term-dependencies

sh run.sh word_to_be_disambiguated number_of_senses_for_the word

e.g., sh run.sh hard 3

interest has 6 senses

line has 6 senses

serve has 4 senses

The script will download gloVe Vectors from https://nlp.stanford.edu/data/glove.6B.zip and train the model, print accuracies and output the incorrect examples in a file.

Note: If you feel the download is taking too much time, download via web browser and comment the wget line in run.sh

 
#Attribution

Datasets used for experiments were from senseval2 competition http://www.senseval.org/data.html

I used glove vectors for intializing word vectors https://nlp.stanford.edu/projects/glove/


Thanks to Dominik inikdom for uploading his code for neural-sentiment https://github.com/inikdom/neural-sentiment  which I used as a starting point.


