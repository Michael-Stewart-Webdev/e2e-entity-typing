# e2e-entity-typing
End-to-end entity typing using Pytorch and BERT.

## Setting up

First place the BERT model folder into `bert/` and start Bert as a Service via the command: 

    bert-serving-start -model_dir cased_L-12_H-768_A-12 -num_worker=1 -max_seq_len=100 -pooling_strategy=NONE
   
Then, open `config.json` and modify it to suit your experiments. The main values to change are:

    "dataset": either "bbn_modified", "figer_50k", or "ontonotes_modified"
    "model": <the name of your model, can be anything>
    "task": either "end_to_end" or "mention_level"
    "embedding_model": either "bert", "glove", or "word2vec"

Please note that if you want to use word2vec or glove you'll need to download the pretrained model files and place them under `/word2vec` and `glove` respectively. The filenames are `glove.6B.300d.txt` (from [here](https://nlp.stanford.edu/projects/glove/)) and `enwiki_20180420_300d.txt` (from [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/)) respectively.

(The datasets are named "bbn_modified", "ontonotes_modified" and "figer_50k" because I took small samples from the test set of the BBN and Ontonotes datasets, and transformed it into a new dataset. The FIGER dataset is the first 50k of the "Wiki" dataset available in the AFET paper.)

## Running the model

First, build the data loaders using:

    python build_data.py
    
The code will transform the relevant dataset in the `/data/datasets` directory into a numerical format. Then, to train the model:

    python train.py

The model will be evaluated during training on the dev set, and then evaluated on the test set once training is complete.