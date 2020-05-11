import numpy as np
from op import *
import op
import gradients
import node
import executor
a=node.Node("a")
b=node.Node("b")
c=matmul(a,b)
print(c.op)

grad_list=gradients.gradients(c,[a,b])
print(len(grad_list))

m=executor.Executor([c]+grad_list)
print(m.run([c]+grad_list,{a:np.array([[2]]),b:np.array([[2]])}))