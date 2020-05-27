import numpy as np
from op import *
import op
import gradients
import node
import executor
import module
import linear
a=node.Node("a")
b=node.Parameter("b")
c=conv2d(a,b)
print(type(b),"b.op")

m=linear.Linear()
for q in m.parameters():
    print(q.const)
grad_list=gradients.gradients(c,[a,b])
for grad in grad_list:
    print(grad.name)
print(len(grad_list))

m=executor.Executor([c]+grad_list)
x=np.ones(1*1*4*4).reshape(1,1,4,4)
y=np.ones(1*1*2*2).reshape(1,1,2,2)
print(m.run({a:x,b:y})[0])

x=np.ones(1*1*2*2).reshape(1,1,2,2)
print(x.strides)
#numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None, subok=False, writeable=True)