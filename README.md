# Audio Classify
This repository implements a CRNN to classify audio clips. 1D convolutions are used to embed sequenced spectrograms, which are fed into a single layer GRU to produce a final embedding for an audio clip. 

This is mostly just a quick project for me to practice some machine learning techniques I didn't to try much of in school, and serve as a starting off point code base for future audio ML projects I'll be doing in the future.

With 150 epochs, this model achieves roughly 85% accuracy when classifying the speakers from the vctk dataset.

## Installation
This project was implemented using pytorch.
```
pip install -r requirements.txt
```

There may be issues with installation order. Check directions in requirements just in case

## Preprocessing
Experiments were performed on the vctk dataset, classifying each audio clip by the speaker. the dataset can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3443). The raw data should be stored in a location specified by the "source" tag in config.toml.

To preprocess the data into a dataframe and store the preprocessed data into a HDF file, run the following command:
```
python preprocessing.py
```


## Train
Training was performed via notebooks on Google colab, but can also be performed via command line via the following command:
```
python train.py
```
I wish I had a cuda enabled GPU to train locally...