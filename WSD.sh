#!/usr/bin/env bash


#download Senseval2 competition datasets and GloVe word vectors
python download_datasets.py

#create vocabulary file
python create_vocab.py

#convert words in data to integer ids to be consumed by training algorithms
python prepare_data.py
#train and print WSD results on different datasets
python train.py
