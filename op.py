import numpy as np
from node import Node
class OP:
    def __call__(self):
        new_node=Node()
        new_node.op=self
        return new_node
    def compute(self,node,vals):
        pass
    def gradient(self,node,grad):
        pass

class AddOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="node_a+node_b"
        return new_node
    def compute(self,node,vals):
        return vals[0]+vals[1]
    def gradient(self,node,grad):
        return [grad,grad]

class MulOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="node_a*node_b"
        return new_node
    def compute(self,node,vals):
        return vals[0]*vals[1]
    def gradient(self,node,grad):
        return [grad*node.parents[1],grad*node.parents[0]]

class ZerosLikeOp(OP):
    def __call__(self,node_a):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.name="zeros(node_a)"
        return new_node
    def compute(self,node,vals):
        return np.zeros(vals[0])
    def gradient(self,node,grad):
        return [zeros_like(node.parents[0])]

class OnesLikeOp(OP):
    def __call__(self,node_a):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.name="Ones(node_a)"
        return new_node
    def compute(self,node,vals):
        return np.ones(vals[0].shape)
    def gradient(self,node,grad):
        return [zeros_like(node.parents[0])]
class PlaceholderOp(OP):
    def __call__(self):
        
        new_node = OP.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        
        return None
add_op=AddOp()
mul_op=MulOp()
zeros_like=ZerosLikeOp()
ones_like=OnesLikeOp()
place_holder=PlaceholderOp()