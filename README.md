# CNN+STFT based Whale Detection Algorithm

So, this repository is my PyTorch solution for the [Kaggle whale-detection challenge](https://www.kaggle.com/c/whale-detection-challenge). The objective of this challenge was to basically do a binary classification, (hence really a detection), on the existance of whale signals in the water. 

It's a pretty cool problem that resonates with prior work I have done in underwater perception algorithm design - a freakishly hard problem I may add. (The speed of sound changes on you, multiple reflections from the environment, but probably the hardest of all being that it's hard to gather ground-truth). (<--- startup idea? :collision: ) 

Anyway! My approach is to first transform the 1D acoustic time-domain signal into a 2D time-frequency representation via the Short-Time-Fourier-Transform (STFT). We do this in the following way:

<img src="https://cloud.githubusercontent.com/assets/27869008/25636131/536fbc9c-2f35-11e7-9669-01e0d98e5d5c.png" width="300">

(Where `K_F` is the raw number of STFT frequency bands, `n` is the discrete time index, `m` is the temporal index of each STFT pixel, `x[n]` the raw audio signal being transformed, and `k` representing the index of each STFT pixel's frequency). In this way, we break the signal down into it's constituent time-frequency energy cells, (which are now pixels), but more crucially, we get a representation that has distinct features across time and frequency that will be correlated with each other. This then makes it ripe for a Convolutional Neural Network (CNN) to chew into. 

Here is what a whale-signal's STFT looks like:

![Pos whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631111/418d17ce-2f24-11e7-91ee-3a1e5e7ed952.png) 

Similarly, here's what a signal's STFT looks like without any whale signal. (Instead, there seems to be some short-time but uber wide band interference at some point in time). 

![Neg whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631319/dc12b5c4-2f24-11e7-8545-7b58950efe99.png)

It's actually interesting, because there are basically so many more ways in which a signal can manifest itself as *not* a whale signal, VS as actually being a whale signal. Does that mean we can also frame the problem as learning the manifold of whale-signals and simply do outlier analysis on that? Something to think about. :) 

# Code Usage:

Ok - let us now talk about how to use the code:

The first thing you need to do is install PyTorch of course. Do this from [here](http://pytorch.org/). I use a conda environment as they recommend, and I recommend you do the same. 

Once this is done, activate your PyTorch environment. 

Now we need to download the raw data. You can get that from Kaggle's site [here](https://www.kaggle.com/c/whale-detection-challenge/data). Unzip this data at a directory of your choosing. For the purpose of this tutorial, I am going to assume that you placed and unzipped the data as such: `/Users/you/data/whaleData/`. (We will only be using the training data so that we can split it into train/val/test. The reason is that we do not have access to Kaggle's test labels).

We are now going to do the following steps:
* Convert the audio files into numpy STFT tensors 
  * `python whaleDataCreatorToNumpy.py -s 1 -dataDir /Users/you/data/whaleData/train/ -labelcsv /Users/you/data/whaleData/train.csv -dataDirProcessed /Users/you/data/whaleData/processedData/ -ds 0.42 -rk 20 200 `
  * The `-s 1` flag says we want to save the results, the `-ds 0.42` says we want to downsample the STFT image by this amount, (to help with computation time), and the `-rk 20 200` says that we want the "rows kept" to be indexed from 20 to 200. This is because the STFT is conjugate symmetric, but also because we make a determination by first **swimming in the data**, that most of the informational content lies between those bands. (Again, the motivation is computational here as well).
* Convert and split the STFT tensors into PyTorch training/val/test Torch tensors.
* Run the CNN classifier!
* Evauluate the results. 



