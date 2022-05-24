# Word2Vec: Learning word embeddings using Skip-gram Method
This repository contains the code for text processing and Naive Bayes Model for multi-class text classification.

### Installation
To install the dependencies run:
```buildoutcfg
pip install -r requirements.txt
```

### Dataset
The [dataset](https://catalog.data.gov/dataset/consumer-complaint-database) is a collection of complaints about consumer financial products and services that we sent to companies for response. The actual text of the complaint by the consumer is given in the `Consumer complaint narrative` column. We are going to use this column to learn the word vectors. 

### Preprocess the data
To do data pre-processing run:
```buildoutcfg
python processing.py
```

### Train the word2vec model
To train the word2vec model run
```buildoutcfg
python Engine.py
```
