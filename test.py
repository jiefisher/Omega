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
class mynet(module.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(filter_shapes=(1,6,5,5),padding=(2,2),stride=(1,1))
        self.pool1 = nn.MaxPool(ksize=(2,2),padding=(0,0),stride=(2,2))
        self.conv2 = nn.Conv2d(filter_shapes=(6,16,5,5),padding=(0,0),stride=(1,1))
        self.pool2 = nn.MaxPool(ksize=(2,2),padding=(0,0),stride=(2,2))
        self.fc1=nn.Linear((16*5*5,120))
        self.fc2=nn.Linear((120,84))
        self.fc3=nn.Linear((84,2))
        # self.relu=relu()
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

fx=node.Node("fx")
fk =node.Node("fk")
re= conv2d(fx,fk)
a= np.ones(1*3*4*4,"float32").reshape(1,3,4,4)
b= np.ones(3*4*2*2,"float32").reshape(3,4,2,2)
grad =gradients.gradients(re,[fx,fk])
e = executor.Executor([re]+grad)
print(e.run(feed_dict={fx:a,fk:b})[1])

labels=node.Node("label")
c = mynet()
x = node.Node("x")
loss = loss.cross_entropy(c(x),labels)
a=[np.ones(1*1*28*28,"float32").reshape(1,1,28,28),np.zeros(1*1*28*28,"float32").reshape(1,1,28,28)]

b=[np.array([1,0]).reshape(1,2),np.array([0,1]).reshape(1,2)]
optimizer=opt.SGD(loss,c.parameters())

for epoch in range(10):
    for batch in range(2):
        
        
        optimizer.step(feed_dict={x:a[batch],labels:b[batch]})
        print(optimizer.parameters[1].const,optimizer.parameters[0].const)

