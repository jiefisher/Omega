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

class SubOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="node_a-node_b"
        return new_node
    def compute(self,node,vals):
        return vals[0]-vals[1]
    def gradient(self,node,grad):
        return [grad,-1*grad]

class AddByConstOp(OP):
    def __call__(self,node_a,const_val):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.const=const_val
        new_node.name="node_a+const"
        return new_node
    def compute(self,node,vals):
        return vals[0]+node.const
    def gradient(self,node,grad):
        return [grad]

class SubByConstOp(OP):
    def __call__(self,node_a,const_val):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.const=const_val
        new_node.name="node_a-const"
        return new_node
    def compute(self,node,vals):
        return vals[0]-node.const
    def gradient(self,node,grad):
        return [grad]

class ConstBySubOp(OP):
    def __call__(self,node_a,const_val):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.const=const_val
        new_node.name="const-node_a"
        return new_node
    def compute(self,node,vals):
        return node.const-vals[0]
    def gradient(self,node,grad):
        return [-1*grad]

class NegOp(OP):
    def __call__(self,node_a):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.name="-node_a"
        return new_node
    def compute(self,node,vals):
        return -1*vals[0]
    def gradient(self,node,grad):
        return [-1*grad]

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

class DivOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="node_a/node_b"
        return new_node
    def compute(self,node,vals):
        return vals[0]/vals[1]
    def gradient(self,node,grad):
        return [grad/node.parents[1],-1*grad*node.parents[0]/node.parents[1]*node.parents[1]]

class MulByConstOp(OP):
    def __call__(self,node_a,const_val):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.const=const_val
        new_node.name="node_b_const"
        return new_node
    def compute(self,node,vals):
        return vals[0]*node.const
    def gradient(self,node,grad):
        return [grad*node.const]

class DivByConstOp(OP):
    def __call__(self,node_a,const_val):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.const=const_val
        new_node.name="node_b/const"
        return new_node
    def compute(self,node,vals):
        return vals[0]/node.const
    def gradient(self,node,grad):
        return [grad/node.const]

class MatMulOp(OP):
    def __call__(self,node_a,node_b,Transpose_A=False,Transpose_B=False):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.Trans_A=Transpose_A
        new_node.Trans_B=Transpose_B
        new_node.name="matmul(node_a,node_b)"
        return new_node
    def compute(self,node,vals):
        if node.Trans_A:
            vals[0]=vals[0].T
        if node.Trans_B:
            vals[1]=vals[1].T
        return np.dot(vals[0],vals[1])
    def gradient(self,node,grad):
        grad_A=matmul(grad,node.parents[1],Transpose_A=False,Transpose_B=True)
        grad_B=matmul(node.parents[0],grad,Transpose_A=True,Transpose_B=False)
        return [grad_A,grad_B]

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

class ReshapeOp(OP):
    def __call__(self,node_a,new_shape):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.new_shape=new_shape
        new_node.name="Reshape(node_a)"
        return new_node
    def compute(self,node,vals):
        return vals[0].reshape(node.new_shape)
    def gradient(self,node,grad):
        return [reshape_grad(node.parents[0])]




class ReshapeGradOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        
        new_node.name="ReshapeGrad(node_a)"
        return new_node
    def compute(self,node,vals):
        return vals[1].reshape(vals[0].shape)
    def gradient(self,node,grad):
        raise "no grad"

class ReduceSumOp(OP):
    def __call__(self,node_a,new_axis):
        new_node=OP.__call__(self)
        new_node.parents=[node_a]
        new_node.new_axis=new_axis
        new_node.name="ReduceSum(node_a)"
        return new_node
    def compute(self,node,vals):
        return np.sum(vals[0], axis=node.new_axis)
    def gradient(self,node,grad):
        return [grad]

class BroadcastToOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="BroadcastTo(node_a,node_b)"
        return new_node
    def compute(self,node,vals):
        return np.broadcast_to(vals[0], shape=vals[1].shape)
    def gradient(self,node,grad):
        grad_A = reduce_sum(grad,axis=0)
        grad_B = zeros_like(node.parents[1])
        return [grad_A,grad_B]

class PlaceholderOp(OP):
    def __call__(self):
        
        new_node = OP.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        
        return None
add_op=AddOp()
sub_op=SubOp()
sub_by_const_op=SubByConstOp()
const_by_sub_op=ConstBySubOp()
neg_op=NegOp()
mul_op=MulOp()
div_op=DivOp()
div_by_const_op=DivByConstOp()
zeros_like=ZerosLikeOp()
ones_like=OnesLikeOp()
place_holder=PlaceholderOp()
add_by_const=AddByConstOp()
mul_by_const=MulByConstOp()
matmul = MatMulOp()
reshape = ReshapeOp()
reshape_grad = ReshapeGradOp()
reduce_sum = ReduceSumOp()
broadcast_to=BroadcastToOp()