# whale-detector

So, this repository is my PyTorch solution for [Kaggle whale-detection challenge](https://www.kaggle.com/c/whale-detection-challenge). The objective of this challenge was to basically do a binary classification (hence really a detection), on the existance of whale signals in the water. 

It's a pretty cool problem that resonates with prior work I have done in underwater perception - a freakishly hard problem I may add. (The speed of sound changes on you, multiple reflections from the environment, but probably the hardest of all being that it's hard to gather ground-truth). (<--- startup idea? :collision: ) 

A power spectrogram of a whale signal is shown below:

![Pos whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631111/418d17ce-2f24-11e7-91ee-3a1e5e7ed952.png) 

Similarly, the power spectrogram of a signal with no whale signal present is shown here:

![Neg whale spectrogram](https://cloud.githubusercontent.com/assets/27869008/25631319/dc12b5c4-2f24-11e7-8545-7b58950efe99.png)
