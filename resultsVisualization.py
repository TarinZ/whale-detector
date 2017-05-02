from helperFunctions import *
from netDefinition import *

# Helper grouping classes.
class directory:
  None
class filename:
  None
class I:
  None    
class N:
  None  


## Actual usage: python resultsVisualization.py -dataDirProcessed /Users/tarinziyaee/data/whaleData/processedData/ -netDir . -imbal 0.8
## Usage: python resultsVisualization.py -t 0 -im 0.77
parser = argparse.ArgumentParser(description='Show test set result.')
parser.add_argument('-dataDirProcessed', dest='dataDirProcessed', required = True, type=str)
parser.add_argument('-netDir', dest='netDir', required = True, type=str)
parser.add_argument('-t', dest='showTestSetResults', default=0)
parser.add_argument('-imbal', dest='negClassDominance', default=0)
args = parser.parse_args()

# Load the validation predictions and targets
valSoftPredictions = np.load('endTrain_valSoftPredictions.npy')
valTargets = np.load('endTrain_valTargets.npy')
accuracies = np.load('endTrain_accuracies.npy')
lossVector = np.load('endTrain_lossVector.npy')
ta_lossVector = np.load('endTrain_ta_lossVector.npy')
va_lossVector = np.load('endTrain_va_lossVector.npy')
valSoftPredictions = np.reshape(valSoftPredictions, (1, -1))
valTargets = np.reshape(valTargets, (1, -1))


# Extract the metrics on the validation set. 
valPrecision, valRecall, valAP, valFPR, valTPR, valAUCROC = extractMetrics(valSoftPredictions, valTargets)
if args.showTestSetResults == '1':
	testSoftPredictions = np.load('endTrain_testSoftPredictions.npy')
	testTargets = np.load('endTrain_testTargets.npy')
	testSoftPredictions = np.reshape(testSoftPredictions, (1, -1))
	testTargets = np.reshape(testTargets, (1, -1))
	testPrecision, testRecall, testAP, testFPR, testTPR, testAUCROC = extractMetrics(testSoftPredictions, testTargets)

# Plots.
fig, ax = plt.subplots(1,2)
ax[0].plot(valRecall.T, valPrecision.T, linewidth=2, marker='.', label='Val');
if args.showTestSetResults == '1':
	ax[0].plot(testRecall.T, testPrecision.T, linewidth=2, marker='.', color='r', label='Test');
ax[0].set_ylim(-0.01, 1.01); ax[0].set_xlim(-0.01, 1.01)
if args.showTestSetResults == '1':
	ax[0].set_title('PRC, valAUPRC=%2.4f, testAUPRC=%2.4f:'%(valAP, testAP))
else:
	ax[0].set_title('PRC, valAUPRC=%2.4f:'%(valAP))
ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision')
ax[0].grid() 
ax[0].legend(loc=0)
ax[1].plot(valFPR.T, valTPR.T, linewidth=2, marker='.', label='Val');
if args.showTestSetResults == '1':
	ax[1].plot(testFPR.T, testTPR.T, linewidth=2, marker='.', color='r', label='Test');
ax[1].set_ylim(-0.01, 1.01); ax[1].set_xlim(-0.01, 1.01)
if args.showTestSetResults == '1':
	ax[1].set_title('ROC, valAUROC=%2.4f, testAUROC=%2.4f'%(valAUCROC, testAUCROC))
else:
	ax[1].set_title('ROC, valAUROC=%2.4f:'%(valAUCROC))
ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR')
ax[1].grid() 

fig, ax = plt.subplots(1,3)
ax[0].plot(accuracies[0,:].T,'-b', linewidth=1, label='Training Accuracy'); 
ax[0].plot(accuracies[1,:].T,'-g', linewidth=1, label='Validation Accuracy'); 
ax[0].set_ylim(50, 100)
ax[0].legend(loc=0)
ax[0].set_title('Accuracies')
ax[0].set_xlabel('VI'); ax[0].set_ylabel('Accuracies')
ax[0].grid() 
ax[1].plot(np.exp(-ta_lossVector[0,:]),'-b',linewidth=1, label='Training Norm-Likelihood'); 
ax[1].plot(np.exp(-va_lossVector[0,:]),'-g',linewidth=1, label='Val Norm-Likelihood'); 
ax[1].set_ylim(0, 1)
ax[1].legend(loc=0)
ax[1].set_title('Likelihoods')
ax[1].set_xlabel('VI'); ax[0].set_ylabel('Likelihoods')
ax[1].grid() 
ax[2].plot(ta_lossVector[0,:],'-b',linewidth=1, label='Training Loss'); 
ax[2].plot(va_lossVector[0,:],'-g',linewidth=1, label='Val Loss'); 
ax[2].set_ylim(0, 1)
ax[2].legend(loc=0)
ax[2].set_title('Losses')
ax[2].set_xlabel('VI'); ax[0].set_ylabel('Loss')
ax[2].grid() 


print ("Val  AUPRC: %2.2f"%(valAP))
if args.showTestSetResults == '1':
	print ("Test AUPRC: %2.2f"%(testAP))

print ("Val  AUROC: %2.2f"%(valAUCROC))
if args.showTestSetResults == '1':
	print ("Test AUROC: %2.2f"%(testAUCROC))


#### To show why ROC is questionable when it comes to imbalanced data:
# From the existing val data set, create an imbalanced version. 
if args.negClassDominance != 0:
	negRatio = float(args.negClassDominance) # Dictate the percentage of negative classes in the data we want to see. 
	I.allPos = np.asarray(np.where(valTargets == 1)[1])
	I.allNeg = np.asarray(np.where(valTargets == 0)[1])
	N.allPos = int((I.allPos).shape[0])
	N.allNeg = int((I.allNeg).shape[0])

	# The new number of positive samples we require, given that all the negative examples will be used. 
	N.sampledPos = int(np.round((N.allNeg/negRatio)*(1-negRatio)))

	# Create the new soft predictions vector
	newSoftValPreds = np.zeros((1, int(N.allNeg + N.sampledPos))).astype(np.float32)

	# Create the new target vector
	newTargets = -1*np.ones((1,int(N.allNeg + N.sampledPos))).astype(np.float32)

	# Randomly chose N.sampledPos indicies from I.appPos	
	I.sampledPos = np.random.choice(I.allPos, N.sampledPos, replace=False)


	# Finally, extract those predictions:	
	newSoftValPreds[0,0:N.sampledPos] = valSoftPredictions[0,I.sampledPos]
	newTargets[0,0:N.sampledPos] = 1
	newSoftValPreds[0,N.sampledPos:] = valSoftPredictions[0,I.allNeg]
	newTargets[0,N.sampledPos:] = 0

	# Extract the new metrics given the imbalanced data.
	newPrecision, newRecall, newAP, newFPR, newTPR, newAUCROC = extractMetrics(newSoftValPreds, newTargets)

	fig, ax = plt.subplots(1,2)
	ax[0].plot(valRecall.T, valPrecision.T, linewidth=2, marker='.', label='Val-Balanced');
	ax[0].plot(newRecall.T, newPrecision.T, linewidth=2, marker='.', color='r', label='Val-Imbalanced');
	ax[0].set_ylim(-0.01, 1.01); ax[0].set_xlim(-0.01, 1.01)	
	ax[0].set_title('PRC, valBalAUPRC=%2.4f, valImbalAUPRC=%2.4f:'%(valAP, newAP))
	ax[0].set_xlabel('Recall'); ax[0].set_ylabel('Precision')
	ax[0].grid() 
	ax[0].legend(loc=0)
	ax[1].plot(valFPR.T, valTPR.T, linewidth=2, marker='.', label='Val-Balanced');	
	ax[1].plot(newFPR.T, newTPR.T, linewidth=2, marker='.', color='r', label='Val-Imbalanced');
	ax[1].set_ylim(-0.01, 1.01); ax[1].set_xlim(-0.01, 1.01)	
	ax[1].set_title('ROC, valBalAUROC=%2.4f, valImbalAUROC=%2.4f'%(valAUCROC, newAUCROC))
	ax[1].set_xlabel('FPR'); ax[1].set_ylabel('TPR')
	ax[1].grid() 

pdb.set_trace()


