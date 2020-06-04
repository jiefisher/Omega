from op import *
import op
import node
import module
import numpy as np
class Linear(module.Module):
    def __init__(self,shapes=(100,10)):
        self.w=node.Parameter("w")
        self.w.const=np.random.randn(shapes[0], shapes[1])/(shapes[0]* shapes[1])
        self.bias=node.Parameter("bias")
        self.bias.const=np.random.randn(1, shapes[1])/(shapes[0]* shapes[1])
    def forward(self,x):
        y1=matmul(x,self.w)
        y2=y1+broadcast_to(self.bias,y1)
        return y2
class Conv2d(module.Module):
    def __init__(self,shapes=(1,5,2,2)):
        self.filters=node.Parameter("filters")
        self.filters.const=np.random.randn(shapes[0], shapes[1],shapes[2], shapes[3])/(shapes[0]* shapes[1]*shapes[2]* shapes[3])
        self.bias=node.Parameter("bias")
        self.bias.const= None 
    def forward(self,x):
        y1=conv2d(x,self.filters)
        if self.bias.const == None:
            self.bias.const=np.random.randn(y1.shape[0], y1.shape[1],y1.shape[2],y1.shape[3])/(y1.shape[0]* y1.shape[1]*y1.shape[2]*y1.shape[3])
        y2=y1+self.bias.const
        return y2
