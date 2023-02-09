# IMPARA

First, prepare the M2 file of the dataset you wish to train.
This is then divided into train/valid/test dataset statement pair.

"m2_to_dataset.py" creates datasets for training QE from these M2 files.
"train_qe.py" trains a BERT model the sentence preferences with created datasets.
