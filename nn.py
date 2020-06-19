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

class rnn(module.Module):
    def __init__(self,batch_size=32,shapes=(100,10),length=1):
        self.w1=node.Parameter("w1")
        self.w1.const=np.random.randn(shapes[0], shapes[1])/(shapes[0]* shapes[1])

        self.w2=node.Parameter("w2")
        self.w2.const=np.random.randn(shapes[0], shapes[1])/(shapes[0]* shapes[1])

        self.w3=node.Parameter("w3")
        self.w3.const=np.random.randn(shapes[0], shapes[1])/(shapes[0]* shapes[1])

        self.s_zero=node.Parameter("s_zero")
        self.s_zero.const=np.random.randn(batch_size, shapes[0])/(batch_size* shapes[0])

        self.length=length
        # self.bias=node.Parameter("bias")
        # self.bias.const=np.random.randn(1, shapes[1])/(shapes[0]* shapes[1])
    def forward(self,x):
        inputs = split(x,nums=self.length,axis=1)
        h=[]
        y=[]
        for i in range(len(inputs)):
            if i==0:
                ht1=self.s_zeros
            else:
                ht1=h[-1]
            ht=matmul(reshape(inputs[i],[1,2]),self.w1)+matmul(ht1,self.w2)
            ht.name="ht"+str(i)
            h.append(ht)
            yt=matmul(ht,self.w3)
            yt.name="yt"+str(i)
            y.append(yt)
        
        return yt

class Conv2d(module.Module):
    def __init__(self,filter_shapes=(1,5,2,2),padding=(0,0),stride=(1,1)):
        self.filters=node.Parameter("filters")
        self.filters.const=np.random.randn(filter_shapes[0], filter_shapes[1],filter_shapes[2], filter_shapes[3])/(filter_shapes[0]* filter_shapes[1]*filter_shapes[2]* filter_shapes[3])
        self.padding = padding
        self.stride = stride
    def forward(self,x):
        y=conv2d(x,self.filters,padding = self.padding,stride = self.stride)
        return y

class MaxPool(module.Module):
    def __init__(self,ksize=(2,2),padding=(0,0),stride=(1,1)):
        self.ksize = ksize
        self.padding = padding
        self.stride = stride
    def forward(self,x):
        y=maxpool(x,self.ksize,padding=self.padding, stride=self.stride)
        return y

class Embedding(module.Module):
    def __init__(self,vocab_size,embedding_dim):
        self.embed_w=node.Parameter("embedding")
        self.embed_w.const=np.random.randn(vocab_size, embedding_dim)/(vocab_size * embedding_dim)
    def forward(self,x):
        y=embed(x,self.embed_w)
        return y

# class RNN(module.Module):
#     def __init__(self,length,state_shape=(0,0)):
#         self.h_state_0=node.Parameter("h_state_0")
#         self.embed_w.const=np.zeros(state_shape)
#     def forward(self,x):
#         a_list = split(x,self.length,1)
