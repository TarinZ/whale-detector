from helperFunctions import *

## Actual usage: python whaleDataCreatorNumpyToTorchTensors.py -numpyDataDir /Users/tarinziyaee/data/whaleData/processedData/ 

# Helper grouping classes.
class directory:
  None
class filename:
  None
class I:
  None    
class N:
  None

# Input parser
parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('-numpyDataDir', dest='numpyDataDir', required = True, type=str)
parser.add_argument('-valPercentage', dest='valPercentage', default=0.2, type=float)
parser.add_argument('-testPercentage', dest='testPercentage', default=0.1, type=float)
args = parser.parse_args()

# Save into local variables
directory.loadNumpyDataFrom = args.numpyDataDir
N.valPercentage = args.valPercentage
N.testPercentage = args.testPercentage

def splitData(testND, valND, trainND, pDataPos, pDataNeg):
  """ This function will operate on binary data, and create a data split of the data set into a training, 
  validation, and test set, whereby the validation and test sets have an equal amount of data per class, 
  but the training set it not guaranteed to. For example, the test set will contain testND positive
  and testND negative samples, the validation set contains valND positives and valND negatives, and 
  the training set contains trainND of the non-dominant class, followed by whatever is left of the 
  dominant class. Lastly, before the training/val/test sets are returned, they are de-meaned and std
  normalized by those factors computed from the training set. 

    Args:
        testND: The non-domimant number of test samples desired.
        valND: The non-domimant number of validation samples desired.
        trainND: The non-domimant number of training samples desired.
        pDataPos: All positive samples from the data set.
        pDataNeg: All negative samples from the data set.


    Returns:        
        pTrainingDataPos: The positive data split for the training data. 
        pTrainingDataNeg: The negative data split for the training data.
        (pValData, pValLabels): The tuple containing the validation data set and validation labels.
        (pTestData, pTestLabels): The tuple containing the test data set and test labels.
        pTrainingMean: The mean image from the training set.
        pTrainingStd: The std image from the training set.        """

  dataShape = pDataPos.shape[1:4]

  # Extract the test set:
  pTestData = np.zeros((2*testND, ) + dataShape).astype(np.float32)
  pTestLabels = -99*np.ones(2*testND).astype(np.int64)
  pTestData[0:testND,:,:,:] = np.copy(pDataPos[0:testND,:,:,:])
  pTestLabels[0:testND] = 1
  pTestData[testND : 2*testND,:,:,:] = np.copy(pDataNeg[0:testND,:,:,:])
  pTestLabels[testND : 2*testND] = 0

  # Extract the validation set:
  pValData = np.zeros((2*valND, ) + dataShape).astype(np.float32)
  pValLabels = -99*np.ones(2*valND).astype(np.int64)
  pValData[0:valND,:,:,:] = np.copy(pDataPos[testND : testND + valND,:,:,:])
  pValLabels[0:valND] = 1
  pValData[valND : 2*valND, :,:,:] = np.copy(pDataNeg[testND : testND + valND,:,:,:])
  pValLabels[valND : 2*valND] = 0

  # Extract the training set, (just split the existing pos/neg splits)
  pTrainingDataPos = np.copy(pDataPos[(testND + valND):, :, :, :])
  pTrainingDataNeg = np.copy(pDataNeg[(testND + valND):, :, :, :])

  # Normalize the data:
  # Compute training mean and std.
  trainingPosNegConcat = np.concatenate((pTrainingDataPos, pTrainingDataNeg), 0)
  pTrainingMean = np.mean(trainingPosNegConcat, 0) 
  pTrainingStd = np.std(trainingPosNegConcat - pTrainingMean, 0)

  # Now de-mean the training and validation sets, using the TRAINING mean of course. 
  pTrainingDataPos -= pTrainingMean 
  pTrainingDataNeg -= pTrainingMean 
  pValData -= pTrainingMean 
  pTestData -= pTrainingMean 

  # Normalize the variance
  pTrainingDataPos /= (pTrainingStd + 1e-6)
  pTrainingDataNeg /= (pTrainingStd + 1e-6)
  pValData /= (pTrainingStd + 1e-6)
  pTestData /= (pTrainingStd + 1e-6)

  return pTrainingDataPos, pTrainingDataNeg, (pValData, pValLabels), (pTestData, pTestLabels), pTrainingMean, pTrainingStd


def minimumSamples(percentage, nNonDominant):
  """ For imbalanced binary data, we still desire a specific percentage split of the total data to go towards training/validation/testing.
  However if data is imbalanced, then we take the minimum number of samples so that are not dominated by the bigger class.

    Args:
        percentage: The percentage of the total data we initially desired to take as a separate split. 
        nNonDominant: The number of non dominant samples.

    Returns:        
        samples: The minimum number of samples commensurate with the desired split and the imbalance.  """  
  samples = np.round(percentage * nNonDominant).astype(np.int64) 
  return samples


# Set the percentage split for validation and test data. 
N.trainPercentage = 1 - (N.testPercentage + N.valPercentage)

# Set the seed used for shuffling the data. 
np.random.seed(1)

# Load all the numpy data
pData = np.load(directory.loadNumpyDataFrom + 'pData' + '.npy') # Data has already been demeaned. 
pLabels = np.load(directory.loadNumpyDataFrom + 'pLabels' + '.npy')

# First shuffle the entire deck
I.randomIndices = np.random.permutation(pData.shape[0]) 
pData = pData[I.randomIndices, :,:,:] 
pLabels = pLabels[I.randomIndices] 

# Split the pData into pDataPos and pDataNeg
pDataPos = np.copy(pData[pLabels==1,:,:,:]) 
pDataNeg = np.copy(pData[pLabels==0,:,:,:]) 

# Determine class dominance:
if pDataPos.shape[0] >= pDataNeg.shape[0]:
  # Negative non-dominant.
  N.nonDominant = pDataNeg.shape[0]
else:
  # Positive non-dominant.
  N.nonDominant = pDataPos.shape[0]

# Compute minimum sample numbers.
N.testND = minimumSamples(N.testPercentage, N.nonDominant) 
N.valND = minimumSamples(N.valPercentage, N.nonDominant)
N.trainND = N.nonDominant - (N.valND + N.testND)

# Create the training/validation/test splits
pTrainingDataPos, pTrainingDataNeg, valTuple, testTuple, _ , _ = splitData(N.testND, N.valND, N.trainND, pDataPos, pDataNeg)

# Save off as torch tensors.
tTrainingDataPos = torch.Tensor(pTrainingDataPos)
tTrainingDataNeg = torch.Tensor(pTrainingDataNeg)
tValData = torch.Tensor(valTuple[0])
tValLabels = torch.Tensor(valTuple[1]).long()
tTestData = torch.Tensor(testTuple[0])
tTestLabels = torch.Tensor(testTuple[1]).long()

## Now save off those tensors:
torch.save(tTrainingDataPos, directory.loadNumpyDataFrom + 'tTrainingDataPos')
torch.save(tTrainingDataNeg, directory.loadNumpyDataFrom + 'tTrainingDataNeg')
torch.save(tValData, directory.loadNumpyDataFrom + 'tValData')
torch.save(tValLabels, directory.loadNumpyDataFrom + 'tValLabels')
torch.save(tTestData, directory.loadNumpyDataFrom + 'tTestData')
torch.save(tTestLabels, directory.loadNumpyDataFrom + 'tTestLabels')

print ('FIN')















