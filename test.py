import numpy as np
from op import *
import gradients
import node
import executor
a=node.Node("a")
b=node.Node("b")
c=a+b
print(c.op)

grad_list=gradients.gradients(c,[a])
print(len(grad_list))

m=executor.Executor([c]+grad_list)
print(m.run([c]+grad_list,{a:2*np.ones(3),b:3*np.ones(3)}))