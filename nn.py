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
        print(y2)
        return y2
class Conv2d(module.Module):
    def __init__(self,filter_shapes=(1,5,2,2),padding=(0,0),stride=(1,1)):
        self.filters=node.Parameter("filters")
        self.filters.const=np.random.randn(filter_shapes[0], filter_shapes[1],filter_shapes[2], filter_shapes[3])/(filter_shapes[0]* filter_shapes[1]*filter_shapes[2]* filter_shapes[3])
        # self.bias=node.Parameter("bias_filter")
        self.padding = padding
        self.stride = stride
        #self.bias.const= np.random.randn(1,1,1,1)
    def forward(self,x):
        y1=conv2d(x,self.filters,padding = self.padding,stride = self.stride)
        # if self.bias.const == np.random.randn(1,1,1,1):
        #     self.bias.const=np.random.randn(y1.shape[0], y1.shape[1],y1.shape[2],y1.shape[3])/(y1.shape[0]* y1.shape[1]*y1.shape[2]*y1.shape[3])
        # y2=y1+self.bias.const
        return y1


class MaxPool(module.Module):
    def __init__(self,ksize=(2,2),padding=(0,0),stride=(1,1)):
        self.ksize = ksize
        self.padding = padding
        self.stride = stride
    def forward(self,x):
        y=maxpool(x,self.ksize,padding=self.padding, stride=self.stride)
        return y