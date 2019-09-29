#!/usr/bin/env bash


#download  GloVe word vector from Stanford NLP website
mkdir data
cd data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
cd .. 


#create vocabulary file
python create_vocab.py $1 $2

#convert words in data to integer ids to be consumed by training algorithms
python prepare_data.py $1 $2

#train and print WSD results 
python train.py $1 $2
