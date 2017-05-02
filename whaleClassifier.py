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


def save_net():  
  net.eval()        
  torch.save(net, 'endTrain_savedNet')

def save_vals_accuracies():
  
  # Pack training and validation results and save off.
  np.save('endTrain_valSoftPredictions', valSoftPredictions)
  np.save('endTrain_valTargets', valTargets)
  np.save('endTrain_accuracies', accuracies)
  

def save_loss_vectors():
  np.save('endTrain_lossVector', lossVector)
  np.save('endTrain_ta_lossVector', ta_lossVector)
  np.save('endTrain_va_lossVector', va_lossVector)

  

###### Usage example: python whaleClassifier.py -g 1 -e 1 -lr 0.002 -L2 0.01 -mb 16 -dp 0 -s 1
## Actual usage: python whaleClassifier.py -dataDirProcessed /Users/tarinziyaee/data/whaleData/processedData/ -g 0 -e 1 -lr 0.0002 -L2 0.01 -mb 4 -dp 0 -s 3 -dnn 'inceptionModuleV1_75x45'
#### Usage with inceptionV1module: python whaleClassifier.py -g 1 -e 1 -lr 5e-5 -L2 0.0001 -mb 6 -dp 0 -s 2
#### Usage with inceptionModuleV1_75x45: python whaleClassifier.py -g 1 -e 10 -lr 5e-5 -L2 0.001 -mb 16 -dp 0 -s 3 -dnn 'inceptionTwoModulesV1_75x45'
#### Usage with inceptionTwoModulesV1_root1_75x45: python whaleClassifier.py -g 1 -e 10 -lr 5e-5 -L2 0.0001 -mb 16 -dp 0 -s 3 -dnn 'inceptionTwoModulesV1_root1_75x45'
parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('-dataDirProcessed', dest='dataDirProcessed', required = True, type=str)
parser.add_argument('-g', dest='gpuFlag', default=0)
parser.add_argument('-dp', dest='dataParallel', default=0)
parser.add_argument('-e', dest='epochs', default=1)
parser.add_argument('-lr', dest='learningRate', default=0.001)
parser.add_argument('-L2', dest='l2_weightDecay', default=0.0001)
parser.add_argument('-mb', dest='minibatchSize', default=16)
parser.add_argument('-s', dest='savingOptions', default=0) #0: None, 1: Only in the end. 2: Every time val-set is run. 3: Save only top performing net. 
parser.add_argument('-dnn', dest='dnnArchitecture', default='inceptionModuleV1_75x45')
parser.add_argument('-nils', dest='numberOfInceptionLayers', default = -1, type=int)
args = parser.parse_args()


directory.loadDataFrom = args.dataDirProcessed
########## PARAMETERS ########################################
N.minibatchSize = int(args.minibatchSize) # Make sure is even since need to do even split with +/- data loaders. (ie, 16 means 8 pos + 8 neg in a minibatch)
N.runValidation = 50 # Will run the validation set and show statistics every x number of batches
N.trainingAccuracySamples = 100 # Number of batches to extract to run the instantaneous training accuracy result on. 
########## DNN definition ########################################

# Make sure mini-batch size is even
if np.mod(N.minibatchSize,2) != 0:
  raise ValueError('Minibatch size must be even.')  

# Create the network
net = Net(args.dnnArchitecture, w_init_scheme = 'He', bias_inits = 1.0, incep_layers = args.numberOfInceptionLayers)


# Settings for if we want to use the GPUs, and/or data-parallelism
if args.gpuFlag != '0':
  torch.cuda.set_device(0)
  net.cuda()  
if args.dataParallel == '1':
  net = torch.nn.DataParallel(net, device_ids=[0,2])

# Setup optimizer
optimizer = optim.Adam(net.parameters(), 
                      lr = float(args.learningRate), 
                      weight_decay = float(args.l2_weightDecay)
                      ) 

############## Load the data ##############
tTrainingDataPos = torch.load(directory.loadDataFrom + 'tTrainingDataPos')
tTrainingDataNeg = torch.load(directory.loadDataFrom + 'tTrainingDataNeg')
tValData = torch.load(directory.loadDataFrom + 'tValData')
tValLabels = torch.load(directory.loadDataFrom + 'tValLabels')
# Training labels are "made" here, only because the torch.utils.data.TensorDataset seems to explicitly need them
tTrainingLabelsPos = torch.ones(tTrainingDataPos.size()[0]).long()
tTrainingLabelsNeg = torch.zeros(tTrainingDataNeg.size()[0]).long()

# Create the TensorDataSets, positive and negative
dataSetPos = torch.utils.data.TensorDataset(tTrainingDataPos, tTrainingLabelsPos)
dataSetNeg = torch.utils.data.TensorDataset(tTrainingDataNeg, tTrainingLabelsNeg)
dataVal = torch.utils.data.TensorDataset(tValData, tValLabels)

# Create the loaders that will shuffle and extract this data for us.
positiveDataLoader = torch.utils.data.DataLoader(dataSetPos, batch_size=N.minibatchSize/2, shuffle=True, num_workers=2)
negativeDataLoader = torch.utils.data.DataLoader(dataSetNeg, batch_size=N.minibatchSize/2, shuffle=True, num_workers=2)
validationDataLoader = torch.utils.data.DataLoader(dataVal, batch_size=N.minibatchSize/2, shuffle=False, num_workers=2)

# More loaders for the computation of the instantenous training accuracy
ins_positiveDataLoader = torch.utils.data.DataLoader(dataSetPos, batch_size=N.minibatchSize/2, shuffle=False, num_workers=2)
ins_negativeDataLoader = torch.utils.data.DataLoader(dataSetNeg, batch_size=N.minibatchSize/2, shuffle=False, num_workers=2)

# Total number of training examples.
N.totalTrainingSamples = tTrainingDataPos.size()[0] + tTrainingDataNeg.size()[0]

# Begin training loop 
N.epochs = int(args.epochs)
N.miniBatchesPerEpoch = int(np.round(N.totalTrainingSamples / float(N.minibatchSize)))
accuracies = np.zeros((2,0)).astype(np.float32)
lossVector = np.zeros((1,0)).astype(np.float32)
ta_lossVector = np.zeros((1,0)).astype(np.float32)
va_lossVector = np.zeros((1,0)).astype(np.float32)
maxValAccuracy = 0.0
sm = torch.nn.Softmax()
for epoch in xrange(N.epochs): 
    
    # Reset the instantaneous loss.
    instantaneousLoss = 0.0
    accumulatedLoss = 0.0

    # Initialize the iterators. 
    posIterator = iter(positiveDataLoader)
    negIterator = iter(negativeDataLoader)

    # Explicitly set the set to training mode. 
    net.train()

    # Perform one epoch-worth of training. An epoch is drawing N.totalTrainingSamples worth of batches. 
    for bb in xrange(N.miniBatchesPerEpoch):

        startTime = time.time()
      	
        # Try to make the iterator for the positive samples increase. If it catches the end,
        # then re-initialize the iterator, and draw once again. 
        try: 
          posBatch, posLabels = posIterator.next()                    
        except:              
          posIterator = iter(positiveDataLoader)
          posBatch, posLabels = posIterator.next()
          
        # Try to make the iterator for the negative samples increase.
        try:
          negBatch, negLabels = negIterator.next()          
        except:
          negIterator = iter(negativeDataLoader)
          negBatch, negLabels = negIterator.next()          

        # At this point have both positive and negative batches and labels.         
        if args.gpuFlag == '0':
          currentBatchData = Variable(torch.cat((posBatch, negBatch), 0), requires_grad=False)
          currentBatchLabels = Variable(torch.cat((posLabels, negLabels), 0), requires_grad=False)
        else:
          currentBatchData = (Variable(torch.cat((posBatch, negBatch), 0), requires_grad=False)).cuda()
          currentBatchLabels = (Variable(torch.cat((posLabels, negLabels), 0), requires_grad=False)).cuda()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
                        
        # Forward prop       
        yEst = net(currentBatchData)
        
        # Compute the loss.
        loss = cross_entropy_loss(yEst, currentBatchLabels)
               
        # Backward prop        
        loss.backward()      

        # Update the weights.         
        optimizer.step()

         # Current instantaneous loss for this iteration. (Not using loss, since want to get it's numerical value only. Variable usage lead to GPU RAM issue.)
        instantaneousLoss = loss.data[0]
        accumulatedLoss += instantaneousLoss
        
        # Save the loss for this minibatch:            
        lossVector = np.append(lossVector, (np.array(instantaneousLoss)).reshape(1,-1), 1).astype(np.float32)   
        
        # Show training & validation statistics
        if bb % N.runValidation == 0: 

            # Store off the average training loss thus far.
            accumulatedLoss /= N.runValidation
               
            # Set the network to eval mode. 
            net.eval()		    
        
            # Loop through training sub-set to compute instantaneous training accuracy
            trainingAccuracy = 0.0      
            trainingCorrect = 0.0
            trainingTotal = 0.0                
            ta_pIter = iter(ins_positiveDataLoader)
            ta_nIter = iter(ins_negativeDataLoader)
            taLoss = 0.0
            
            iterationAccumulator = 0.0
            for tt in xrange(N.trainingAccuracySamples):
            
              try:                
                ta_posData, ta_posLabels = ta_pIter.next()          
              except:            
                break

              ta_negData, ta_negLabels = ta_nIter.next()
              ta_totalData = torch.cat((ta_posData, ta_negData), 0)
              ta_totalLabels = torch.cat((ta_posLabels, ta_negLabels), 0)

              if args.gpuFlag == '0':
                ta_totalData, ta_totalLabels = Variable(ta_totalData, requires_grad=False), Variable(ta_totalLabels, requires_grad=False)
              else:
                ta_totalData, ta_totalLabels = (Variable(ta_totalData, requires_grad=False)).cuda(), (Variable(ta_totalLabels, requires_grad=False)).cuda()

              # Forward prop.
              yEst_ta = net(ta_totalData)

              # Extract the predictions:
              # Get the soft predictions for the positive-class
              ta_softPredictions = sm(yEst_ta)[:,1]
            
              # Compute the loss            
              taLoss += cross_entropy_loss(yEst_ta, ta_totalLabels).data[0]


              trainingCorrect += ((ta_softPredictions.cpu().data.numpy() > 0.5) == ta_totalLabels.data.cpu().numpy().T).sum()
              trainingTotal += ta_totalLabels.size(0)

              iterationAccumulator += 1

            # Append the training loss.
            taLoss /= iterationAccumulator
            ta_lossVector = np.append(ta_lossVector, (np.array(taLoss)).reshape(1,-1), 1).astype(np.float32)        
            trainingAccuracy = 100.0 * trainingCorrect / float(trainingTotal)            

            # Compute validation predictions, losses, and accuracies.            
            valSoftPredictions, valTargets, vaLoss= extractForwardPropResults_binary(net, validationDataLoader, gpuFlag=args.gpuFlag)            

            # pdb.set_trace()
            valAccuracy = 100.0 * np.sum((valSoftPredictions>0.5)==valTargets) / valTargets.shape[0]                    
            va_lossVector = np.append(va_lossVector, (np.array(vaLoss)).reshape(1,-1), 1).astype(np.float32)                    

            # Append allaccuracies                    
            accuracies = np.append(accuracies, [[trainingAccuracy], [valAccuracy]], 1)            
            print('[%d, %5d] [Ins Training loss: %.3f] [Accum Training loss: %.3f] [TA Training loss: %.3f] [Val loss: %.3f] [Training Accuracy: %2.2f] [Val Accuracy: %2.2f]' % (epoch+1, bb+1, instantaneousLoss, accumulatedLoss, taLoss, vaLoss, trainingAccuracy, valAccuracy))
            
            
            ##### Save off the results thus far: ######
            if args.savingOptions == '2':

              # Save the net, validation outputs/targets, and accuracies.
              save_net_vals_accuracies()
              
              # Save off the loss vectors.
              save_loss_vectors()
            
            elif args.savingOptions == '3':
              # Save off the loss vectors.
              save_loss_vectors()

              # Save the val accuracies
              save_vals_accuracies()  

              # Want to save off the net and corresponding vectors, only if the net has the highest validation score thus far. 
              if valAccuracy > maxValAccuracy:

                # Save the net, validation outputs/targets, and accuracies.
                save_net()
                maxValAccuracy = valAccuracy
            
            # End timer
            endTime = time.time()
            totalTime = endTime - startTime
            print ("Time: %2.2f"%(totalTime))

            # Explicitly set the set the net to go back to training mode. 
            net.train()


print('Finished Training')


# Save off net and statistics if saving option is enabled.
if args.savingOptions == '1' or args.savingOptions == '2':
  
  # Save off the loss vectors.
  save_loss_vectors()
 
  # Extract validation statistics
  valSoftPredictions, valTargets = extractForwardPropResults(net, validationDataLoader,  gpuFlag=args.gpuFlag)
  
  # Save the net
  save_net()

  # Save the val accuracies
  save_vals_accuracies()  

print ("FIN")
pdb.set_trace()



