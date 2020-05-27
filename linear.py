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
        y=matmul(x,self.w)
        y=y+broadcast_to(self.bias,y)
        return y