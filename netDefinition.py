from dependencies import *
from helperFunctions import *


class Net(nn.Module):
  """
  The class that defines the graph architecture of the DNN to be used. Definition of convolutional
  layers, fully connected layers, batch-normalization operations, etc are created here. 

  Args:
    None: Network definition within the initialization routine. 

    Returns:
        Instantiates a Net object, with member function forward_prop. The forward_prop member function
        input [x] takes in the input signal, and returns the output signal at the output of the net. """        

  def __init__(self, arch, w_init_scheme = 'He', bias_inits = 0.0, incep_layers = -1, nEmbed = -1, nClasses = -1):
    super(Net, self).__init__()
 
    # Initialization scheme
    self.weightsInitsScheme = w_init_scheme
    self.biasesInitTo = bias_inits
    self.N_incep_layers = incep_layers
    # Save off the selected DNN arch as a member. 
    self.arch = arch    

    # Dictionary of supported DNN architectures.    
    switcher = {'cnn_108x108': self.cnn_108x108, 
                'inceptionModuleV1_108x108': self.inceptionModuleV1_108x108,
                'inceptionModuleV1_75x45': self.inceptionModuleV1_75x45,
                'inceptionTwoModulesV1_75x45': self.inceptionTwoModulesV1_75x45,
                'inceptionTwoModulesV1_root1_75x45': self.inceptionTwoModulesV1_root1_75x45,
                'inceptionV1_modularized': self.inceptionV1_modularized,
                'inceptionV1_modularized_mnist': self.inceptionV1_modularized_mnist,
                'centerlossSimple': self.centerlossSimple
               }
    

    # Select the net definition given by arch.    
    netDefinition = switcher.get(arch)
    
    # Initialize the architecture selected.     
    try:
        if self.arch == 'centerlossSimple':
            assert(nEmbed > 0)
            assert(nClasses > 0)
            netDefinition(nEmbed, nClasses)
        else:
            netDefinition()
        
        print "DNN arch: ", self.arch
    except:
        print 'Specified DNN architecture not implemented.'
        sys.exit()

    # How the percentage of each layer's trainable parameters as a 
    # function of the total number of trainable parameters    
    self.show_layer_parameter_percentages()
            
    # Initialize the DNN with the specified scheme.
    self.initialize_layers()

  



  ######################################## Member functions ########################################
  def show_layer_parameter_percentages(self):
    
    # Compute the total number of learnable parameters:
    paramIterator = list(self.parameters())
    self.N_dnnParameters = 0
    for pp in paramIterator:                
        if len(pp.size()) == 1:
            self.N_dnnParameters += pp.size()[0]
        else:
            self.N_dnnParameters += np.prod(pp.size())


    # Show number of learnable parameters as function of module.
    self.N_runningParams = 0        
    for m in self.modules():        
        if isinstance(m, Net):            
            continue
        elif isinstance(m, nn.Sequential):
            print '\n'
            continue        
        elif isinstance(m, nn.ModuleList):
            print '\n'
            continue                    
        else:
            params = list(m.parameters())
            N_lenParams = len(params)
            N_currentParams = 0            
            if N_lenParams > 0:                
                for pp in xrange(N_lenParams):
                    N_currentParams += np.prod(params[pp].size())
            else:
                N_currentParams = 0

            self.N_runningParams += N_currentParams                
            print ('Module: ' + m.__class__.__name__ + ' params: %2.2f')%(100.0*N_currentParams / float(self.N_dnnParameters))

    print ('\n')    
    print ('Total number of trainable parameters: %5d\n')%(self.N_dnnParameters)
    assert(self.N_runningParams == self.N_dnnParameters)        
    

  def initialize_layers(self):
    
    print ('Initializing layers via: ' + self.weightsInitsScheme + ', biases to: %2.2f')%(self.biasesInitTo)
    
    # Initialize selected layers
    for m in self.modules():        
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
          
            # Compute the fan in.   
            if isinstance(m, nn.Conv2d):
                fanIn = m.in_channels * m.weight.size()[2] * m.weight.size()[3]
            elif isinstance(m, nn.Linear):
                fanIn = m.in_features

            # Print the layer:        
            print ('Module: ' +  m.__class__.__name__ + ' FanIn: %1d')%(fanIn)

            # Perform initialization on the weights.        
            if self.weightsInitsScheme == 'He':    
                m.weight.data.normal_(0, np.sqrt(2.0 / fanIn))

            # Perform initializatio of the biases
            try:
                m.bias.data.fill_(self.biasesInitTo)
            except:
                print "No biases found for this layer."



  # Definition of the forward prop              
  def forward(self, x):

    if self.arch == 'cnn_108x108':
        x = self.cnn(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return x

    elif self.arch == 'inceptionModuleV1_108x108' or self.arch == 'inceptionModuleV1_75x45':                
        root = self.root(x)
        b_1x1 = self.b_1x1(root)
        b_3x3 = self.b_3x3(root)
        b_5x5 = self.b_5x5(root)
        b_pool = self.b_pool(root)
        concat = torch.cat( (b_1x1, b_3x3, b_5x5, b_pool), 1)                
        redux = self.redux(concat)        
        redux = redux.view(-1, self.num_flat_features(redux))        
        logits = self.classifier(redux)        
        return logits

    elif self.arch == 'inceptionTwoModulesV1_75x45':
        root = self.root(x)

        b_1x1 = self.b_1x1(root)
        b_3x3 = self.b_3x3(root)
        b_5x5 = self.b_5x5(root)
        b_pool = self.b_pool(root)
        concat = torch.cat( (b_1x1, b_3x3, b_5x5, b_pool), 1)      

        b2_1x1 = self.b2_1x1(concat)
        b2_3x3 = self.b2_3x3(concat)
        b2_5x5 = self.b2_5x5(concat)
        b2_pool = self.b2_pool(concat)
        concat2 = torch.cat( (b2_1x1, b2_3x3, b2_5x5, b2_pool), 1)              
        
        redux = self.redux(concat2)        
        redux = redux.view(-1, self.num_flat_features(redux))        
        logits = self.classifier(redux)        

        return logits        


    elif self.arch == 'inceptionTwoModulesV1_root1_75x45':

        root = self.root(x)
        
        b_1x1 = self.b_1x1(root)
        b_3x3 = self.b_3x3(root)
        b_5x5 = self.b_5x5(root)
        b_pool = self.b_pool(root)
        concat = torch.cat( (b_1x1, b_3x3, b_5x5, b_pool), 1)      
        
        b2_1x1 = self.b2_1x1(concat)
        b2_3x3 = self.b2_3x3(concat)
        b2_5x5 = self.b2_5x5(concat)
        b2_pool = self.b2_pool(concat)
        concat2 = torch.cat( (b2_1x1, b2_3x3, b2_5x5, b2_pool), 1)              
                
        redux = self.redux(concat2)        
        redux = redux.view(-1, self.num_flat_features(redux))        
        logits = self.classifier(redux)        

        return logits        

    elif (self.arch == 'inceptionV1_modularized') or (self.arch =='inceptionV1_modularized_mnist'):
    
        # Forward prop through the root first.
        # pdb.set_trace()
        root = self.root(x)

        # Loop through each each inception layer
        for ii in xrange(self.N_incep_layers):
            
            # If processing the next inception layer, the new 'root' signal is the inception output (incepOut) of the previous layer.
            if ii > 0:
                root = incepOut

            # Loop through each branch of the current inception layer.             
            for bb in xrange(len(self.masterList[ii])):                                
                temp = self.masterList[ii][bb](root)
                if bb == 0:
                    incepOut = temp
                else:
                    incepOut = torch.cat((incepOut, temp), 1)

        # Forward through the redux layer.                    
        redux = self.redux(incepOut)                    
        redux = redux.view(-1, self.num_flat_features(redux))       

        # Forward through the final fully connected layer.
        self.x = self.fc1(redux)

        # Save intermediate x
        # self.x = x.cpu().data.numpy()

        # reld = self.r1(self.x)
        # logits = self.fc2(reld)
        logits = self.fc2(self.x)
        

        # pdb.set_trace()
        # Return logits        
        return logits            


    elif self.arch == 'centerlossSimple':
    
        root = self.root(x)
        root = root.view(-1, self.num_flat_features(root))       
        self.x = self.latent(root)
        logits = self.logits(self.x)
        
        return logits        








  # Helper function to flatten an input vector from 
  def num_flat_features(self, x):    
    # all dimensions except the batch dimension
    size = x.size()[1:] 
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


  def centerlossSimple(self, nEmbed, nClasses):

    
    # Initialize the centroids:
    self.centroids = torch.from_numpy(0.01*np.random.randn(nEmbed, nClasses)).cuda(0)    

    # The root
    self.root = nn.Sequential(
        nn.Conv2d(1,32, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.Conv2d(32,32, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 32x14x14     

        # input: 32x14x14
        nn.Conv2d(32, 64, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 64x7x7

        # input: 64x7x7
        nn.Conv2d(64, 128, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 5, stride = (1,1), padding = (2,2), bias=True),        
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 128x3x3
        )    

    # input: 128x3x3        
    self.latent = nn.Linear(1152, nEmbed, bias=True)        
    # output: nEmbed x 1

    # input: nEmbedx1
    self.logits = nn.Linear(nEmbed, nClasses, bias=False)
    # output: nClasses x 1


  """inceptionV1_modularized_mnist """
  def inceptionV1_modularized_mnist(self):

    # Initialize the centroids:
    self.centroids = torch.from_numpy(np.random.randn(2, 10)).cuda(0)    
    # self.cGradients = torch.zeros(2,10).cuda(0)
    # self.centroids = Variable(torch.from_numpy(np.random.randn(2, 10)), requires_grad=False ).cuda(0)    
    # self.centroids = torch.random(2,10)

    # pdb.set_trace()
    assert(isinstance(self.N_incep_layers, int))    
    if self.N_incep_layers <= 0:
        print "Selected inceptionV1_modularized, but number of inception layers wanted is less than 0."
        sys.exit()

    # Root layers
    self.root = nn.Sequential(        
        # input: 1x28x28
        nn.Conv2d(1,64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 1x14x14     

        # input: 1x14x14
        nn.Conv2d(64, 256, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)        
        # output: 256x14x14
        )

    # Inception Chains.
    self.masterList = nn.ModuleList()
    for ii in xrange(self.N_incep_layers):        
        incep = nn.ModuleList()
        incep += self.create_inception_module_v1()
        self.masterList += [incep]

   
    # Redux layers
    self.redux = nn.Sequential(
        # input: 256x14x14      
        nn.Conv2d(256, 64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),        
        # output: 64x14x14      

        # input: 64x14x14
        nn.Conv2d(64, 32, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # # output: 32x7x7

        # input: 32x7x7
        nn.Conv2d(32, 16, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 16x3x3

        # input: 16x3x3
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace = True)
        # output: 4x3x3        
        )         

    self.fc1 = nn.Linear(36, 2)        

    self.fc2 = nn.Linear(2, 10, bias=False)        

    return None 




  """inceptionV1_modularized """
  def inceptionV1_modularized(self):

    # pdb.set_trace()
    assert(isinstance(self.N_incep_layers, int))    
    if self.N_incep_layers <= 0:
        print "Selected inceptionV1_modularized, but number of inception layers wanted is less than 0."
        sys.exit()

    # Root layers
    self.root = nn.Sequential(        
        # input: 1x75x45        
        nn.Conv2d(1,64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 1x37x22     

        # input: 1x37x22 
        nn.Conv2d(64, 256, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)        
        # output: 192x18x11
        )

    # Inception Chains.
    self.masterList = nn.ModuleList()
    for ii in xrange(self.N_incep_layers):        
        incep = nn.ModuleList()
        incep += self.create_inception_module_v1()
        self.masterList += [incep]

   
    # Redux layers
    self.redux = nn.Sequential(
        # input: 256x37x22      
        nn.Conv2d(256, 64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),        
        # output: 64x18x11      

        # input: 64x18x11
        nn.Conv2d(64, 32, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # # output: 32x9x5

        # input: 32x9x5
        nn.Conv2d(32, 16, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 16x4x2

        # input: 16x4x2
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace = True)
        # output: 4x4x2        
        )         

    self.classifier = nn.Sequential(
        # input: 1x32
        nn.Linear(180, 2)
        # output: 1x2
        )

    return None 

    
  """create_inception_module_v1"""    
  def create_inception_module_v1(self):
    # First inception v1 module
    b_1x1 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(256, 64, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x37x22 
        )
        
    b_3x3 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(256, 96, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x37x22 
        )        

    b_5x5 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(256, 16, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )        

    b_pool = nn.Sequential(
        # input: 192x37x22 
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(256, 32, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )        

    # Combine the inception branches into a list and return. List consumed by nn.ModuleList()
    return [b_1x1, b_3x3, b_5x5, b_pool]
    


  """Definition of the cnn_108x108 arch."""
  def cnn_108x108(self):    

    self.cnn = nn.Sequential(
        
        nn.Conv2d(1, 8, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(8),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),

        nn.Conv2d(8, 16, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),

        nn.Conv2d(16, 32, 3, stride = (1,1), padding = (0,0)),        
        nn.Dropout2d(),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),

        nn.Conv2d(32, 64, 3, stride = (1,1), padding = (1,1)),                
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),

        nn.Conv2d(64, 64, 3, stride = (3,3), padding = (0,0)),                
        nn.Dropout2d(),
        nn.ReLU(inplace=True)                            
        )

    self.classifier = nn.Sequential(
        nn.Linear(256, 32),
        nn.Dropout(),
        nn.Linear(32, 16),
        nn.Linear(16, 2)
        )

  # Definition of the inceptionModuleV1_108x108 arch.
  def inceptionModuleV1_108x108(self):
    
    self.root = nn.Sequential(
        
        # input: 1x108x108
        nn.Conv2d(1, 64, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),

        nn.Conv2d(64, 192, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2))
        # output: 192x27x27
    )

    self.b_1x1 = nn.Sequential(
        # input: 192x27x27
        nn.Conv2d(192, 64, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x27x27
        )
        
    self.b_3x3 = nn.Sequential(
        # input: 192x27x27
        nn.Conv2d(192, 96, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x27x27
        )        

    self.b_5x5 = nn.Sequential(
        # input: 192x27x27
        nn.Conv2d(192, 16, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x27x27
        )        

    self.b_pool = nn.Sequential(
        # input: 192x27x27
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(192, 32, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x27x27
        )        

    self.redux = nn.Sequential(
        # input: 256x27x27        
        nn.Conv2d(256, 64, 2, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 64x14x14

        # input: 64x14x14
        nn.Conv2d(64, 32, 1, stride = (1,1), padding = (0,0)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 32x7x7

        # input: 32x7x7
        nn.Conv2d(32, 16, 1, stride = (1,1), padding = (0,0)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((3,3), stride = (2,2), padding = (0,0)),
        # output: 16x3x3

        # input: 16x3x3
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0)),
        nn.ReLU(inplace = True)
        # output: 4x3x3        
        )         

    self.classifier = nn.Sequential(
        # input: 1x36
        nn.Linear(36, 2)
        # output: 1x2
        )


# Definition of the inceptionModuleV1_75x45 arch.
  def inceptionModuleV1_75x45(self):
    
    # pdb.set_trace()
    self.root = nn.Sequential(
        
        # input: 1x75x45        
        nn.Conv2d(1, 64, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 1x37x22 

        # input: 1x37x22 
        nn.Conv2d(64, 192, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2))
        # output: 192x18x11
    )

    self.b_1x1 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 64, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x18x11
        )
        
    self.b_3x3 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 96, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x18x11
        )        

    self.b_5x5 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 16, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )        

    self.b_pool = nn.Sequential(
        # input: 192x18x11
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(192, 32, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )        

    self.redux = nn.Sequential(
        # input: 256x18x11      
        nn.Conv2d(256, 64, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),        
        # output: 64x18x11      

        # input: 64x18x11
        nn.Conv2d(64, 32, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # # output: 32x9x5

        # input: 32x9x5
        nn.Conv2d(32, 16, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 16x4x2

        # input: 16x4x2
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0)),
        nn.ReLU(inplace = True)
        # output: 4x4x2        
        )         

    self.classifier = nn.Sequential(
        # input: 1x32
        nn.Linear(32, 2)
        # output: 1x2
        )




  # Definition of the inceptionTwoModulesV1_75x45 arch.
  def inceptionTwoModulesV1_75x45(self):
    
    # pdb.set_trace()
    self.root = nn.Sequential(
        
        # input: 1x75x45        
        nn.Conv2d(1, 64, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 1x37x22 

        # input: 1x37x22 
        nn.Conv2d(64, 192, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2))
        # output: 192x18x11
    )


    # First inception v1 module
    self.b_1x1 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 64, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x18x11
        )
        
    self.b_3x3 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 96, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x18x11
        )        

    self.b_5x5 = nn.Sequential(
        # input: 192x18x11
        nn.Conv2d(192, 16, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )        

    self.b_pool = nn.Sequential(
        # input: 192x18x11
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(192, 32, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )        



    # Second inception v1 module
    self.b2_1x1 = nn.Sequential(
        # input: 256x18x11
        nn.Conv2d(256, 64, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x18x11
        )
        
    self.b2_3x3 = nn.Sequential(
        # input: 256x18x11
        nn.Conv2d(256, 96, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1)),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x18x11
        )        

    self.b2_5x5 = nn.Sequential(
        # input: 256x18x11
        nn.Conv2d(256, 16, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )        

    self.b2_pool = nn.Sequential(
        # input: 256x18x11
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(256, 32, 1, stride = (1,1), padding = (0,0)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x18x11
        )     


    self.redux = nn.Sequential(
        # input: 256x18x11      
        nn.Conv2d(256, 64, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),        
        # output: 64x18x11      

        # input: 64x18x11
        nn.Conv2d(64, 32, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # # output: 32x9x5

        # input: 32x9x5
        nn.Conv2d(32, 16, 3, stride = (1,1), padding = (1,1)),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 16x4x2

        # input: 16x4x2
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0)),
        nn.ReLU(inplace = True)
        # output: 4x4x2        
        )         

    self.classifier = nn.Sequential(
        # input: 1x32
        nn.Linear(32, 2)
        # output: 1x2
        )




  # Definition of the inceptionTwoModulesV1_root1_75x45 arch.
  def inceptionTwoModulesV1_root1_75x45(self):
        
    self.root = nn.Sequential(
        
        # input: 1x75x45        
        nn.Conv2d(1,64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2,2), stride=(2,2)),
        # output: 1x37x22     

        # input: 1x37x22 
        nn.Conv2d(64, 192, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True)        
        # output: 192x18x11
    )

    # First inception v1 module
    self.b_1x1 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(192, 64, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x37x22 
        )
        
    self.b_3x3 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(192, 96, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x37x22 
        )        

    self.b_5x5 = nn.Sequential(
        # input: 192x37x22 
        nn.Conv2d(192, 16, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )        

    self.b_pool = nn.Sequential(
        # input: 192x37x22 
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(192, 32, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )        



    # Second inception v1 module
    self.b2_1x1 = nn.Sequential(
        # input: 256x37x22 
        nn.Conv2d(256, 64, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        # output: 64x37x22 
        )
        
    self.b2_3x3 = nn.Sequential(
        # input: 256x37x22 
        nn.Conv2d(256, 96, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(96),        
        nn.ReLU(inplace=True),
        nn.Conv2d(96, 128, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
        # output: 128x37x22 
        )        

    self.b2_5x5 = nn.Sequential(
        # input: 256x37x22 
        nn.Conv2d(256, 16, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, 5, stride = (1,1), padding = (2,2), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )        

    self.b2_pool = nn.Sequential(
        # input: 256x37x22 
        nn.MaxPool2d((3,3), stride = (1, 1), padding = (1,1)),          
        nn.Conv2d(256, 32, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        # output: 32x37x22 
        )     


    self.redux = nn.Sequential(
        # input: 256x37x22      
        nn.Conv2d(256, 64, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),        
        # output: 64x18x11      

        # input: 64x18x11
        nn.Conv2d(64, 32, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # # output: 32x9x5

        # input: 32x9x5
        nn.Conv2d(32, 16, 3, stride = (1,1), padding = (1,1), bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace = True),
        nn.MaxPool2d((2,2), stride = (2,2)),
        # output: 16x4x2

        # input: 16x4x2
        nn.Conv2d(16, 4, 1, stride = (1,1), padding = (0,0), bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace = True)
        # output: 4x4x2        
        )         

    self.classifier = nn.Sequential(
        # input: 1x32
        nn.Linear(180, 2)
        # output: 1x2
        )








