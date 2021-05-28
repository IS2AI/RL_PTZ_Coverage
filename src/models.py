'''
Duelling networks
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuellingModelCnn(nn.Module):
    """
    Deep Q network model for 1 agent
    """
    def __init__(self):

        """
        Params:
        """

        super(DuellingModelCnn, self).__init__()
        self.conv1 = nn.Conv2d(2,16,7,3)
        self.conv2 = nn.Conv2d(16,32,5,2)
        
        self.conv3 = nn.Conv2d(32,64,3,1)
        #self.conv4 = nn.Conv2d(16,32,4,2)

        #self.pool = nn.MaxPool2d(2, 2)

        self.fc1v = nn.Linear(87616,512)
        self.fc1v1 = nn.Linear(512,512)
        self.fc2v = nn.Linear(512,1)

        self.fc1a = nn.Linear(87616,512)
        self.fc1a1 = nn.Linear(512,512)
        self.fc2a = nn.Linear(512,6)

    def forward(self, x):
        """ Run the network"""
            
        x = F.relu(self.conv1(x))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        #x = self.pool(x)           
         
        x = x.view(x.size(0),-1)
        #print('flatten: ', x.shape) 
        #exit()
         
        v = F.relu(self.fc1v(x))
        v = F.relu(self.fc1v1(v))
        v = self.fc2v(v)
        
        a = F.relu(self.fc1a(x))
        a = F.relu(self.fc1a1(a))
        a = self.fc2a(a)

        q = v + a - a.mean(dim=1, keepdim=True)

        return q


  
'''
Actor Critic CNN -- Ping Pong
'''
class ActorCriticCnn(nn.Module):

    def __init__(self):
        super(ActorCriticCnn, self).__init__()
        # ptz
        self.conv1 = nn.Conv2d(2,16,7,3)
        self.conv2 = nn.Conv2d(16,32,5,2)
        self.conv3 = nn.Conv2d(32,64,3,1)

        self.fc1 = nn.Linear(87616, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc_a = nn.Linear(512, 6)
        self.fc_c = nn.Linear(512, 1)

        self.sf = nn.Softmax(dim=1)

    def forward(self, x):

        #print('1 ', x.shape)
        x = torch.unsqueeze(x, axis=0)
        #print('2 ', x.shape)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0),-1)

        #print('flatten: ', x.shape) 

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x1 = self.fc_a(x)
        x1 = self.sf(x1)

        x2 = self.fc_c(x)

        x1 = torch.squeeze(x1)

        x2 = torch.squeeze(x2)
        #print(x1, x2)

        return x1, x2

