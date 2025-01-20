import torch
import math
import copy
from Atiny import Atiny

maxTimes=100000
BatchSize=1000

size=64
layers=20
features=512

Adam_lr=5e-4
Atiny_lr=2e-4
weight_decay=0


class Block(torch.nn.Module):
    def __init__(self,in_features,out_features,gain=1,**kwargs):
        super(Block, self).__init__()
        self.Linear=torch.nn.Linear(in_features,out_features,bias=True,**kwargs)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.Linear.weight,gain)
            self.Linear.bias.fill_(0)
        
    def forward(self,input):
        return self.Linear(input).asinh()
    

#This network architecture is also my original work and follows the same open source license,
#including its extension to any/every dimensional convolution and replacement of any/every activation functions.        

class AccessNet(torch.nn.Module):
    def __init__(self,in_features,out_features,hidden_layers,hidden_features,**kwargs):
        super(AccessNet, self).__init__()
        self.inLayer=torch.nn.Linear(in_features,hidden_features,bias=True,**kwargs)
        self.Blocks=torch.nn.Sequential() 
        for t in range(hidden_layers):
            self.Blocks.add_module('hiddenLayer{0}'.format(t),Block(hidden_features,hidden_features,**kwargs))
        self.outLayer=torch.nn.Linear(hidden_features,out_features,bias=True,**kwargs)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.inLayer.weight,1)
            self.inLayer.bias.fill_(0)
            torch.nn.init.orthogonal_(self.outLayer.weight,1/math.sqrt(hidden_layers))
            self.outLayer.bias.fill_(0)
    def forward(self,input):
        step=self.inLayer(input)
        outputAccess=torch.zeros_like(step)
        for block in self.Blocks:
            step=block(step)
            outputAccess=outputAccess+step
        output=self.outLayer(outputAccess)
        return output

device = "cuda" if torch.cuda.is_available() else "cpu"

module1=AccessNet(size,size,layers,features,device=device)
module2=copy.deepcopy(module1)
optimizer1 = torch.optim.Adam(module1.parameters(),Adam_lr,weight_decay=weight_decay)
optimizer2 = Atiny(module2.parameters(),lr = Atiny_lr,weight_decay=weight_decay)

index=torch.arange(0,2*size,1,device=device,dtype=torch.float32).unsqueeze(0)

import visdom
wind = visdom.Visdom(env="Optimizer Test", use_incoming_socket=False)
    
wind.line([[float('nan'),float('nan')]],[0],win = 'loss',opts = dict(title = 'log(loss)/log(batchs)',legend = ['Adam','Atiny']))
wind.line([[float('nan'),float('nan')]],[0],win = 'contrastAdam',opts = dict(title = 'Adam: Comparison of curve prediction results',legend = ['output','target']))
wind.line([[float('nan'),float('nan')]],[0],win = 'contrastAtiny',opts = dict(title = 'Atiny: Comparison of curve prediction results',legend = ['output','target']))

print(module2)
    
for time in range(maxTimes):
    a=torch.randn(BatchSize,device=device).unsqueeze(1)
    b=torch.randn(BatchSize,device=device).unsqueeze(1)
    c=torch.randn(BatchSize,device=device).unsqueeze(1)*size
    input,target=torch.chunk(((((index*a+c).sin()+1)/2)**b.exp())-(((1-((index*a+c).sin()))/2)**(-b).exp()),chunks=2,dim=1)
    
    module1.zero_grad()
    output=module1(input)
    loss1=torch.nn.functional.mse_loss(output,target)
    loss1.backward()
    optimizer1.step()
    
    L1=[output[0].tolist(),target[0].tolist()]
    L1=list(map(list, zip(*L1)))
    

    module2.zero_grad()
    output=module2(input)
    loss2=torch.nn.functional.mse_loss(output,target)
    loss2.backward()
    optimizer2.step()
    
    L2=[output[0].tolist(),target[0].tolist()]
    L2=list(map(list, zip(*L2)))
    

    wind.line(L1,win = 'contrastAdam',opts = dict(title = 'Adam: Comparison of curve prediction results',legend = ['output','target']))
    wind.line(L2,win = 'contrastAtiny',opts = dict(title = 'Atiny: Comparison of curve prediction results',legend = ['output','target']))
    wind.line([[float(loss1.log()),float(loss2.log())]],[math.log(time+1)],win = 'loss',update = 'append')

    