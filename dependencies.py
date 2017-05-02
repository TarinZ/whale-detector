
from __future__ import print_function
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset
import sys
import matplotlib
import h5py
import numpy
import numpy as np
import os
import random
import cPickle as pkl
import datetime
import time
import gzip
from six.moves import urllib
from six.moves import xrange  # pylint: disabl
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import glob
import fnmatch
import md5
import hashlib
import aifc
import scipy
from scipy.fftpack import fft
import csv
import cPickle as pickle 
import matplotlib.patches as patches

# If LINUX OS
if sys.platform == "linux" or sys.platform == "linux2":
  # TODO: Add matplotlib viz specifics
  print ("LINUX OS")
  
# If Mac OSX
elif sys.platform == "darwin":
  print ("MAC OSX")  
  matplotlib.use('TkAgg')
  import matplotlib.pyplot as plt
  plt.ion()
  
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


