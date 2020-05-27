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