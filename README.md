# whale-detector

So, this repository is my PyTorch solution for the [Kaggle whale-detection challenge](https://www.kaggle.com/c/whale-detection-challenge). The objective of this challenge was to basically do a binary classification, (hence really a detection), on the existance of whale signals in the water. 

It's a pretty cool problem that resonates with prior work I have done in underwater perception algorithm design - a freakishly hard problem I may add. (The speed of sound changes on you, multiple reflections from the environment, but probably the hardest of all being that it's hard to gather ground-truth). (<--- startup idea? :collision: ) 

Anyway! My approach is to first transform the 1D acoustic time-domain signal into a 2D time-frequency representation via the Short-Time-Fourier-Transform (STFT). We do this in the following way:

![eq1](https://cloud.githubusercontent.com/assets/27869008/25635293/7c1a98e0-2f32-11e7-9c9d-fa0179143217.png width = "100")

In this way, we break the signal down into it's constituent time-frequency energy cells, (which are now pixels), but more crucially, we get a representation that has distinct features across time and frequency that will be correlated with each other. This then makes it ripe for a Convolutional Neural Network (CNN) to chew into. 

Here is what a whale-signal's STFT looks like:

![Pos whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631111/418d17ce-2f24-11e7-91ee-3a1e5e7ed952.png) 

Similarly, here's what a signal's STFT looks like without any whale signal. (Instead, there seems to be some short-time but uber wide band interference at some point in time). 

![Neg whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631319/dc12b5c4-2f24-11e7-8545-7b58950efe99.png)

It's actually interesting, because there are basically so many more ways in which a signal can manifest itself as *not* a whale signal, VS as actually being a whale signal. Does that mean we can also frame the problem as learning the manifold of whale-signals and simply do outlier analysis on that? Something to think about. :) 
