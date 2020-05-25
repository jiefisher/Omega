import numpy as np
from op import *
import op
import gradients
import node
import executor
a=node.Node("a")
b=node.Node("b")
c=conv2d(a,b)
print(c.op)

grad_list=gradients.gradients(c,[a,b])
for grad in grad_list:
    print(grad.name)
print(len(grad_list))

m=executor.Executor([c]+grad_list)
x=np.ones(1*1*4*4).reshape(1,1,4,4)
y=np.ones(1*1*2*2).reshape(1,1,2,2)
print(m.run({a:x,b:y})[1].shape)