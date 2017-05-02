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
* Convert the audio files into numpy STFT tensors: 
  * `python whaleDataCreatorToNumpy.py -s 1 -dataDir /Users/you/data/whaleData/train/ -labelcsv /Users/you/data/whaleData/train.csv -dataDirProcessed /Users/you/data/whaleData/processedData/ -ds 0.42 -rk 20 200 `
  * The `-s 1` flag says we want to save the results, the `-ds 0.42` says we want to downsample the STFT image by this amount, (to help with computation time), and the `-rk 20 200` says that we want the "rows kept" to be indexed from 20 to 200. This is because the STFT is conjugate symmetric, but also because we make a determination by first **swimming in the data**, (I swear this pun is not intentional), that most of the informational content lies between those bands. (Again, the motivation is computational here as well).
* Convert and split the STFT tensors into PyTorch training/val/test Torch tensors:
  * `python whaleDataCreatorNumpyToTorchTensors.py -numpyDataDir /Users/you/data/whaleData/processedData/ `
  * Here, the original numpy tensors are first split and normalized, and then saved off into PyTorch tensors. (The split percentages are able to be user defined, I set the defaults set 20% for validation and 10% test). The PyTorch tensors are saved in the same directory as above.
* Run the CNN classifier!
  * We are now ready to train the classifier! I have already designed an Inception-V1 CNN architecture, that can be loaded up automatically, and we can use this as so. The input dimensions are also guaranteed to be equal to the STFT image sizes here. At any rate, we do this like so:
  * `python whaleClassifier.py -dataDirProcessed /Users/you/data/whaleData/processedData/ -g 0 -e 1 -lr 0.0002 -L2 0.01 -mb 4 -dp 0 -s 3 -dnn 'inceptionModuleV1_75x45'`
  * The `g` term controls whether or not we want to use a GPU to trian, `e` controls the number of epochs we want to train over, `lr` is the learning rate, `L2` is the L2 penalization amount for regularization, `mb` is the minibatch size, (which will be double this as the training composes a mini-batch to have an equal number of positive and negative samples), `dp` controls data parallelism (moot without multiple GPUs, and is really just a flag on whether or not to use multiple GPUs), `s` controls when and how often we save the net weights and validation losses, (option `3` saves the best performing model), and finally, `-dnn` is a flag that controls which DNN architecture we want to use. In this way, you can write your own DNN arch, and then simply call it by whatever name you give it for actual use. (I did this after I got tired of hard-coding every single DNN I designed). 
  * If everything is running smoothly, you should see something like this as training progresses: <img src="https://cloud.githubusercontent.com/assets/27869008/25638321/dd562a16-2f3c-11e7-99c5-e1f0392bbdf7.png" width="600">
  * The "time" here just shows how long it takes between the reporting of each validation score. (Since I ran this on my CPU, it's 30 seconds / report, but expect this to be at least an order of magnitude faster on a respectable GPU).
* Evauluate the results! 
  * When your training is complete, you can then then run this script to give you automatically generated ROC and PR curves for your network's performance:
  * `python resultsVisualization.py -dataDirProcessed /Users/you/data/whaleData/processedData/ -netDir . `
  * After a good training session, you should get results that look like so:
  * <img src="https://cloud.githubusercontent.com/assets/27869008/25638818/7c13e08e-2f3e-11e7-9f08-5fcc028b2b59.png" width="600">
  * I also show the normalized training / validation likelihoods and accuracies for the duration of the session:
  * <img src="https://cloud.githubusercontent.com/assets/27869008/25638967/0e0bf242-2f3f-11e7-9f18-a23f61fae2f2.png" width="600">






