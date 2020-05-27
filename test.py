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
a=np.array([2*np.ones(1*4),np.ones(1*4)])
b=np.array([np.array([1,0]),np.array([0,1])])
optimizer=opt.SGD(loss,c.parameters())
optimizer.step(feed_dict={x:a,labels:b})
print(optimizer.parameters[1].const)
#numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)