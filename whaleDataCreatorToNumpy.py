from helperFunctions import *


## Simple & fast usage          : python whaleDataCreatorToNumpy.py -s 1 -dataDir /Users/you/data/whaleData/train/ -labelcsv /Users/you/data/whaleData/train.csv -dataDirProcessed /Users/you/data/whaleData/processedData/
## Simple & fast usage with viz : python whaleDataCreatorToNumpy.py -s 1 -ins 2 -dataDir /Users/you/data/whaleData/train/ -labelcsv /Users/you/data/whaleData/train.csv -dataDirProcessed /Users/you/data/whaleData/processedData/ 
## Actual usage                 : python whaleDataCreatorToNumpy.py -s 1 -dataDir /Users/tarinziyaee/data/whaleData/train/ -labelcsv /Users/tarinziyaee/data/whaleData/train.csv -dataDirProcessed /Users/tarinziyaee/data/whaleData/processedData/ -ds 0.42 -rk 20 200 
## Advanced usage               : python whaleDataCreatorToNumpy.py -s 1 -dataDir /Users/you/data/whaleData/train/ -labelcsv /Users/you/data/whaleData/train.csv -dataDirProcessed /Users/you/data/whaleData/processedData/ -fs 2000.0 -tx 2.0 -tf 0.071 -po 0.75 -fftl 512 -fftw 'hanning' -ds 0.42 -rk 20 200 

def id_duplicates(dir):
    """ Function to detect duplicate files in the trianing data.
    Args:
        dir: Absolute directory of the training files.
    Returns:        
        dupes: List of the duplicate file names to be skipped. """
    unique = []
    filehash = []
    dupes = []
    for filename in os.listdir(dir):        
        if os.path.isfile(dir+filename):
            filehash = md5.md5(file(dir+filename).read()).hexdigest()            
        if filehash not in unique: 
            unique.append(filehash)
        else:             
            dupes.append(filename)
    return dupes

# Helper grouping classes.
class directory:
  None
class filename:
  None
class F:
  None
class T:
  None
class N:
  None    
class I:
  None      


## Parsers
# Required
parser = argparse.ArgumentParser(description='Settings')
parser.add_argument('-dataDir', dest='dataDir', required = True, type=str)
parser.add_argument('-labelcsv', dest='labelcsv', required = True, type=str)
parser.add_argument('-dataDirProcessed', dest='dataDirProcessed', required = True, type=str)
# Optional
parser.add_argument('-fs', dest='samplingRateHz', default=2000.0, type=float)
parser.add_argument('-tx', dest='timeSamplesPerExample', default=2.0, type=float)
parser.add_argument('-tf', dest='timeSamplesPerFrame', default=0.071, type=float)
parser.add_argument('-po', dest='percentageOverlapPerFrame', default=0.75, type=float)
parser.add_argument('-fftl', dest='fftLength', default=512, type=int)
parser.add_argument('-fftw', dest='fftWindow', default='rect')
parser.add_argument('-rk', dest='rowsKept', default=(20,128), nargs='+', type=int)
parser.add_argument('-ds', dest='downsampleImage', default= -1.0, type=float) # -1: No. Otherwise, fraction will be factor of downsampling. (eg, 0.4, 0.7, etc)
parser.add_argument('-s', dest='savingOptions', default=0, type=int) # 0: No, 1: Yes. 
parser.add_argument('-ins', dest='inspectAndPause', default=0, type=int) # 0: None, 1: Pause on first sample, 2: show images as processed
args = parser.parse_args()

# Set the physical parameters of the data:
directory.dataDir = args.dataDir
filename.labelcsv = args.labelcsv
directory.dataDirProcessed = args.dataDirProcessed
F.fs = args.samplingRateHz # Sampling rate. [Hz]
T.x = args.timeSamplesPerExample # Time extent of each training data. [s]
T.frameLength = args.timeSamplesPerFrame # Desired frame time extent. [s]
T.olap = args.percentageOverlapPerFrame*T.frameLength # Desired overlap time extent. [s]
N.fftLength = args.fftLength # Desired FFT length of the STFT matrix. [bins]
I.rowsKept =  np.asarray(range( (args.rowsKept)[0] , (args.rowsKept)[1])) # Indicies of the positive frequencies to excise.
fftWindow = args.fftWindow #'hanning' # The FFT windowing equation to utilize. 

# Create the STFT transformer object
stftObj = STFT(F.fs, T.x, T.olap, T.frameLength, fftLength=N.fftLength, window=fftWindow, flagDebug = True)

# Read the training labels.
with open(filename.labelcsv, 'rb') as f:    
    reader = csv.reader(f)
    csvList = list(reader)  
    csvList = csvList[1:]
N.data = len(csvList)

# Extract the data and save off into numpy arrays first...
if args.downsampleImage != -1:  
  pData = np.zeros((N.data, 1, int(np.floor(args.downsampleImage*len(I.rowsKept))), int(np.floor(args.downsampleImage*stftObj.N.frames)) )).astype(np.float32)
else:
  pData = np.zeros((N.data, 1, int(len(I.rowsKept)), int(stftObj.N.frames))).astype(np.float32)
pLabels = -1*np.ones(N.data).astype(np.int64)

# ID the duplicate files.
dupes = id_duplicates(directory.dataDir)

# Look through the CSV label file, skip dupes, and process each training file, to convert into the STFT matrix. 
cc = 0
for ii in xrange(N.data):    

  # The current file to process:
  filename.currentTrainingFile = directory.dataDir + csvList[ii][0]

  # Check if the file is duplicated, if it is, skip it. 
  if csvList[ii][0] in dupes:
    print ("[DUPE]: ", filename.currentTrainingFile)
    continue
  else:
    
    # Extract the STFT image and place into pData  
    fileHandle = aifc.open(filename.currentTrainingFile, mode='r')
    audioString = fileHandle.readframes(fileHandle.getnframes())
    signal = (numpy.fromstring(audioString, numpy.short).byteswap()).astype(np.float32)
    
    # De-mean the audio signal:
    signal -= np.mean(signal)
    # Divide by the std of the audio signal to normalize it's variance to unity. 
    signal /= np.std(signal)

    # TODO: force audio signal to take on values between -1 and 1)
    # Take this data file's short-time-fourier-transform (STFT)
    stftObj.computeSTFT(signal)
    stftImage = np.abs((stftObj.stftMatrix)[I.rowsKept,:])

    # Downsample the STFT image (optional)
    if args.downsampleImage != -1.0:
      stftImage = scipy.misc.imresize(stftImage, args.downsampleImage, interp='bicubic'); 

    # Place processed STFT image into pData array.    
    pData[cc, 0, :, :] = stftImage
  
    # Extract label and place into pLabels
    pLabels[cc] = int(csvList[ii][1])
    
    # Inspect and pause (optional)
    if args.inspectAndPause == 1:
      pdb.set_trace()    
    elif args.inspectAndPause == 2:            
      plt.cla()
      plt.imshow(pData[cc, 0, :, :], interpolation='None', aspect='auto'); plt.show()  
      plt.title(['ii: ', str(ii), ' label: ', str(pLabels[cc])], fontsize=20, fontweight='bold')
      plt.pause(0.2)
      # raw_input()

    # Update the counter
    cc += 1

    # [OK] file processed. 
    print '[OK]: ' + filename.currentTrainingFile 

# Excise the extra amounts, since the detected dupes were not processed.
pData = pData[0:cc]
pLabels = pLabels[0:cc]

if args.savingOptions == 1:
  # Save the numpy arrays to file
  print ('Saving pData to disk...')
  np.save(directory.dataDirProcessed + 'pData', pData)
  print ('Saving pLabels to disk...')
  np.save(directory.dataDirProcessed + 'pLabels', pLabels)


print ("FIN")
