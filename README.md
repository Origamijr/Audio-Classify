# Audio Classify
stub readme

## Installation
```
pip install -r requirements.txt
```

There may be issues with installation order. Check directions in requirements just in case

## Preprocessing
```
python preprocessing.py
```

## Train
```
python train.py
```

## Notes

1/14/22 - Opted for 1D convolution over 2D convolution. 1D convolution overall decreases the size of the network, and I figured spectrograms don't carry to much locality in the frequency axis compared to the time axis. Experimented with various activations and model normalization techniques. Tried SELU, but training was far too unstable at the moment, may need further investigation into ideal training conditions and benefits in inference time. Both batch normalization and layer normalization seem to provide relatively stable training losses. Some anomalies occur in training, so may need to investigate learning rate adjustments, or other optimization parameters. Current progress: validation accuracy ~20% loss ~3.5