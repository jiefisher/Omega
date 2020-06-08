import numpy as np
from op import *
import op
import gradients
import node
import executor
class Optimizer:
    def __init__(self, cost,parameters):
        assert parameters, 'Your parameters?'
        self.parameters = list(parameters)
        self.cost = cost
        self.grads = gradients.gradients(self.cost,self.parameters)
        self.exe=executor.Executor([self.cost]+self.grads)

    # def zero_grad(self):
    #     for x in self.grads_val:
    #         #p.grad = np.zeros_like(p.data)
    #         x *= 0.0

    def step(self, lr=0.01):
        assert False, 'Optimizer class is virtual'


class SGD(Optimizer):
    def step(self, feed_dict,lr=0.001):
        output = self.exe.run(feed_dict)
        for i in range(len(self.parameters)):
            self.parameters[i].const-=lr*output[i+1]
        
