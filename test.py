import numpy as np
from op import *
import op
import gradients
import node
import executor
import module
import linear
import loss
import opt
class mynet(module.Module):
    def __init__(self,shapes):
        self.shapes = shapes
        self.fc=linear.Linear(self.shapes)
    def forward(self,x):
        y=self.fc(x)
        return y

labels=node.Node("label")
c = mynet(shapes=(4,2))
x = node.Node("x")
loss = loss.cross_entropy(c(x),labels)
a=[2*np.ones(1*4).reshape(1,4),np.ones(1*4).reshape(1,4)]
b=[np.array([1,0]).reshape(1,2),np.array([0,1]).reshape(1,2)]
for epoch in range(10):
    for batch in range(2):
        optimizer=opt.SGD(loss,c.parameters())
        
        optimizer.step(feed_dict={x:a[batch],labels:b[batch]})
print(optimizer.parameters[1].const)