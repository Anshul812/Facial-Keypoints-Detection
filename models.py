
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
       
        self.conv1 = nn.Conv2d(1, 32, 7, 3, 1)       
        self.conv2 = nn.Conv2d(32, 64, 5, 3, 0)      
        self.conv3 = nn.Conv2d(64, 128, 5, 3, 1)     
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 0)    
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 0)    
        self.conv6 = nn.Conv2d(512, 512, 1, 1, 0)    
        
        self.fc1 = nn.Linear(4*4*512, 1024)
        self.fc1_drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc2_drop = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(1024, 136)
        
    def forward(self, x):
        # Apply convolutional layers
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = F.selu(self.conv5(x))
        x = F.selu(self.conv6(x))

        # Flatten and continue with dense layers
        x = x.view(x.size(0), -1)
        x = F.selu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.selu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        return x



net = gpu(Net())
print(net)

