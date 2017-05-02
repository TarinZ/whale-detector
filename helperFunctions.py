from dependencies import *

class STFT(object):
	"""
    Computes the Short-Time-Fourier-Transform (STFT) of a signal. 

    Args:
        fs: Sampling rate. [Hz]
        xT: Length of the input signal. [samples]
        overlapT: Time to overlap frames by. [s]
        frameT: The time extent of each frame. [s]
        fftLength: The length of the FFT to take. [bins]
        flagDebug: Flag on prints for debug mode. [boolean]
        window: A specification of what pre-FFT windowing equation to use [sting], eg. 'hanning'

    Returns:
        Will internally compute and store the positive frequency excised STFT matrix, in 
        self.stftMatrix. """		

	def __init__(self, fs, xT, overlapT, frameT, fftLength=None, flagDebug=False, window='rect'):	

		# Helper grouping classes.
		class F:
			None
		class T:
			None
		class N:
			None
								
		# Helper class for frequencies.										
		self.F = F
		# Helper class for time extents.		
		self.T = T
		# Helper class for sample numbers. 
		self.N = N

		# Sampling rate. [Hz]
		self.F.fs = fs 

		# Length of time of the input x to be STFT'd. [s]
		self.T.x = xT

		# Length of time in overlap of frames. [s]
		self.T.olap = overlapT

		# Length of time of each STFT frame desired. [s]
		self.T.frame = frameT

		# Compute the number of samples in one frame. [samples]
		self.N.frameLength = np.floor(self.F.fs * self.T.frame).astype(np.int64)

		# Compute the number of samples in the input [samples]
		self.N.x = np.floor(self.F.fs * self.T.x)

		# Compute the number of samples for overlap. 
		self.N.olap = np.floor(self.F.fs * self.T.olap)

		# Compute the FFT length to be used. [samples]
		if fftLength == None:
			# If not specified, then use an FFT length equal to the length of each frame.
			self.N.fft = self.N.frameLength
		else:
			# If the FFT length is specified, then use that. 
			if fftLength < self.N.frameLength:
				print ("FFT length given smaller than frame length. Defaulting to FFT length equal to frame length.")
				self.N.fft = self.N.frameLength
			else:
				self.N.fft = fftLength
		self.N.fft = self.N.fft

		# Since the FFT is redundant, compute want the 0Hz to (Fs/2 - delta_f) frequency range.  
		# Assuming python indexing:
		# For positive freqencies, want indices to be [0 : ceil(Nfft/2.0) - 1] (inclusive)
		# For negative frequenies, want indicies to be [ceil(N/2.0) : N - 1] (inclusive)
		# For this STFT implementation, we only report the positive frquencies. 
		self.N.stftRows = np.floor(np.ceil(self.N.fft / 2.0))

		# Compute the stride. [samples]
		self.N.stride = self.N.frameLength - self.N.olap

		# Based on N.frameLength, N.x, and N.olap, compute the number of frames (columns) of the STFT
		self.N.frames = np.floor((self.N.x - self.N.frameLength) / (self.N.stride).astype(np.float32)) + 1.0

		# Allocate memory for the STFT matrix (complex valued)			
		self.stftMatrix = np.zeros(((self.N.stftRows).astype(np.int64), (self.N.frames).astype(np.int64))) .astype(np.complex64)

		# Select the window to use
		if window=='rect':
			self.window = np.ones(1,self.N.frameLength).astype(np.float32)
		elif window=='hanning':
			nn = np.arange(0,self.N.frameLength)			
			self.window = 0.5*(1 - np.cos( (2.0 * np.pi * nn) / (self.N.frameLength - 1))).astype(np.float32)
		else:
			print('Window not specified.')
			sys.exit(1)
					
		# Interal print outs if in debug mode.					
		if flagDebug == True:
			print ("N.x          : ", self.N.x)			
			print ("N.frameLength: ", self.N.frameLength)
			print ("N.fft        : ", self.N.fft)
			print ("N.olap       : ", self.N.olap)
			print ("N.stride     : ", self.N.stride)			
			print ("N.stftRows   : ", self.N.stftRows)
			print ("N.frames     : ", self.N.frames)
					
	def computeSTFT(self, x):
		""" Computes the positive-frquency modulus of the excised STFT matrix. """
		indiciesVector = np.asarray(range(0 , self.N.frameLength)).astype(np.int64)
		for ff in xrange((self.N.frames).astype(np.int64)):
			# Compute the STFT				
			self.stftMatrix[:,ff] = fft(x[indiciesVector]*self.window, self.N.fft)[0:(self.N.stftRows).astype(np.int64)]
			# Compute the indices into x, into which we extract a frame to FFT
			indiciesVector += (self.N.stride).astype(np.int64)


def extractForwardPropResults_binary(theNet, theDataLoader, gpuFlag='0'):
  """ Loads a saved neural network and forward propagates the input signal through to attain the 
  final multinomial logistic output.

  Args:
    theNet: The neural net that has already been trained and loaded. 
    theDataLoader: The Torch data loader object already associated with the loading directory.
    gpuFlag: string flag '0' or '1' indicating whether or not the neural net was created on GPU RAM.

  Returns:        
    softPredictions: The probability of the positive target being present post-detection through the net.
    targets: The ground truth targets.
    lossValue: The average loss value across all training samples & labels provided. """

  
  lossValue = 0.0
  softPredictions = np.zeros((1,0)).astype(np.float32)
  targets = np.zeros((1,0)).astype(np.float32)  

  # Begin looping through the data loader provided. (One epoch)
  for ff, data in enumerate(theDataLoader, 0):
    
    # Extract a validation batch    
    images, labels = data
    
    # Case into torch Variables, on either CPU or GPU.
    if gpuFlag=='0':
      images, labels = Variable(images, requires_grad=False), Variable(labels, requires_grad=False)
    else:    
      images, labels = (Variable(images, requires_grad=False)).cuda(), (Variable(labels, requires_grad=False)).cuda()

    # Forward prop the batch        
    yEst = theNet(images) 

    # Accumulate the losses.     
    lossValue += cross_entropy_loss(yEst, labels, gpuFlag).data[0]
    
    # Get the soft predictions for the positive-class
    softPreds = sm(yEst)[:,1]

    # Concatenate the soft predictions.    
    softPredictions = np.append(softPredictions, softPreds.data.cpu().numpy()).astype(np.float32)

    # Concatenate the targets:
    targets = np.append(targets, labels.data.cpu().numpy().astype(np.float32)).astype(np.float32)

  # Normalize the loss  
  lossValue /= ff

  return softPredictions, targets, lossValue

  
def info(data):
  """ Helper function that gives information on numpy arrays.
    Prints a quick summary of numpy array type, shape, max and min.

    Args:
        input: The numpy array.

    Returns:        
        None """
        
  # If is a torch tensor          
  if str(type(data))[0:14] == "<class 'torch.":
    print data.size(), ' Torch', type(data).__name__, ' Max: ', torch.max(data), ' ', ' Min: ', torch.min(data)
  elif str(type(data))[0:13] == "<type 'numpy.":   
    print data.shape, ' ', type(data).__name__, ' ', data.dtype, ' Max: ', np.max(data), ' ', ' Min: ', np.min(data)
  return ''

def qImage(image):
	""" Helper function to quickly image 2D data of arbitrary matrix size.	
	Args:
	    input: The numpy array to be images.

	Returns:        
	    None """
  	plt.imshow(image, interpolation='None', aspect='auto'); plt.show()

	
"""Robust Torch softmax function handle"""
sm = torch.nn.Softmax()  


# def cross_entropy_loss(logits, labels, indexToClass = None):
def cross_entropy_loss(logits, labels, gpuFlag='0'):
  """ Helper function to compute cross entropy loss. 
  Args:
    logits: The logits output by the classifier, (given as torch.FloatTensor Variable), eg, a 4x2 matrix, 4 is the minibatch size, across 2 classes.
    labels: The labels, (given as torch.LongTensor Variable). eg, a 4x1 vector, where each element encodes the correct index of the class. In a 2 class problem, would therfore be either 0 or 1.     

  Returns:        
    loss: The cross-entropy loss. """        
  
  eps = 1e-6

  # Probabilities        
  probs = sm(logits)

  # pdb.set_trace()
  if gpuFlag=='0':
    selectedProbs = torch.LongTensor(int(probs.size()[0]))
  else:
    selectedProbs = torch.LongTensor(int(probs.size()[0])).cuda()
  
    
  # Select the proper probability indices from the targets.
  selectedProbs = probs.gather(1, labels.view(-1, 1))

  # Compute the cross-entropy loss
  loss = torch.mean(-torch.log(selectedProbs + eps))

  return loss


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    imageSize = 28
    numChannels = 1
    pixelDepth = 255
    bytestream.read(16)
    buf = bytestream.read(imageSize * imageSize * num_images * numChannels)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (pixelDepth / 2.0)) / pixelDepth
    data = data.reshape(num_images, imageSize, imageSize, numChannels)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels






