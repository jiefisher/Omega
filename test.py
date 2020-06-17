import numpy as np
from op import *
import op
import gradients
import node
import executor
import module
import nn
import loss
import opt
from activation import  *



val=np.array([[0.1,0.2,0.3],[0.1,0.2,0.3]]).reshape(2,3)
s = val.reshape(-1,1)

t = np.diagflat(s) - np.dot(s, s.T)
origin_shape=val.shape
val = val.reshape(1,-1)
res= np.dot(val,t)
res=res.reshape(origin_shape)
print(res)
print(res.shape)
print(res)
print(t.shape,val.shape,np.diagflat(s).shape,np.dot(s, s.T).shape)

em = np.array([[0.1, 0.2, 0.3, 0.4],
     [0.5, 0.6, 0.7, 0.8],
     [0.9, 0.0, 0.1, 0.2]])

inx=np.array([0, 2, 1, 2]).reshape(1,4)


class embnet(module.Module):
    def __init__(self):
        self.embed = nn.Embedding(3,4)
        self.embed.embed_w.const = em
        self.fc3=nn.rnn((20,2))
    def forward(self,x):
        y1=self.embed(x)
        y2=reshape(y1,[-1,4*5])
        y = self.fc3(y2)
        return y
x = node.Node("x")
labels=node.Node("label")

c = embnet()


eloss = loss.cross_entropy(c(x),labels)
a=[np.array([0, 2, 1, 2,1]).reshape(1,5),np.array([0, 2, 1, 2,2]).reshape(1,5)]
b=[np.array([1,0]).reshape(1,2),np.array([0,1]).reshape(1,2)]
optimizer=opt.SGD(eloss,c.parameters())

for epoch in range(10):
    for batch in range(2):
        optimizer.step(feed_dict={x:a[batch],labels:b[batch]})
        print(optimizer.parameters[1].const,optimizer.parameters[0].const)
exit()
class mynet(module.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(filter_shapes=(1,6,5,5),padding=(2,2),stride=(1,1))
        self.pool1 = nn.MaxPool(ksize=(2,2),padding=(0,0),stride=(2,2))
        self.conv2 = nn.Conv2d(filter_shapes=(6,16,5,5),padding=(0,0),stride=(1,1))
        self.pool2 = nn.MaxPool(ksize=(2,2),padding=(0,0),stride=(2,2))
        self.fc1=nn.Linear((16*5*5,120))
        self.fc2=nn.Linear((120,84))
        self.fc3=nn.Linear((84,2))
    def forward(self,x):
        y1=self.conv1(x)
        y2=self.pool1(y1)
        y3=self.conv2(y2)
        y4=self.pool2(y3)
        y4=reshape(y4,[-1,16*5*5])
        y5 = relu(self.fc1(y4))
        y6 = self.fc2(y5)
        y = self.fc3(y6)
        return y

x = node.Node("x")
labels=node.Node("label")

cnn = mynet()


loss = loss.cross_entropy(cnn(x),labels)
a=[np.ones(1*1*28*28,"float32").reshape(1,1,28,28),np.zeros(1*1*28*28,"float32").reshape(1,1,28,28)]
b=[np.array([1,0]).reshape(1,2),np.array([0,1]).reshape(1,2)]
optimizer=opt.SGD(loss,cnn.parameters())


for epoch in range(10):
    for batch in range(2):
        optimizer.step(feed_dict={x:a[batch],labels:b[batch]})
        # print(optimizer.parameters[1].const,optimizer.parameters[0].const)

