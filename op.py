import numpy as np
from node import Node

gres= np.ones(1)
gidx= {}
n=0
dx_shape= {}
dX_col = np.ones(1)
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
class EmbedOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="embed(node_a)"
        return new_node
    def compute(self,node,vals):
        max_value=np.max(vals[0])+1
        vals_eye=np.eye(max_value)[vals[0]]
        t= np.dot(vals_eye.reshape(-1,vals[1].shape[0]),vals[1])

        return t.reshape(vals[0].shape[0],t.shape[0],t.shape[1])
    def gradient(self,node,grad):
        return [None,embed_grad(node.parents[0],node.parents[1],grad)]

class EmbedGradientOp(OP):
    def __call__(self,node_a,node_b,node_c):
        #node_a:input
        #node_b:embed_param
        #node_c:grad
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b,node_c]
        new_node.name="embed_grad(node_a)"
        return new_node
    def compute(self,node,vals):
        vals[2] = vals[2].reshape(vals[2].shape[0]*vals[2].shape[1],vals[2].shape[2])
        max_value=np.max(vals[0])+1
        vals_eye=np.eye(max_value)[vals[0]]
        t= np.dot(vals_eye.reshape(vals[1].shape[0],-1),vals[2])

        return t
    def gradient(self,node,grad):
        raise "no grad"

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
        t=  np.dot(vals[0],vals[1])
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
        return np.zeros(vals[0].shape)
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
        if node.new_shape[0]==-1:
            node.new_shape[0]=vals[0].shape[0]
        print(vals[0].shape,node.new_shape)
        return vals[0].reshape(node.new_shape)
    def gradient(self,node,grad):
        return [reshape_grad(node.parents[0],grad)]




class ReshapeGradOp(OP):
    def __call__(self,node_a,node_b):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        
        new_node.name="ReshapeGrad(node_a,node_b)"
        return new_node
    def compute(self,node,vals):
        return vals[1].reshape(vals[0].shape)
    def gradient(self,node,grad):
        print("aa")
        raise NotImplementedError('Gradient of ReshapeGradientOp not supported')

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
    def __call__(self,node_a,node_b,):
        new_node=OP.__call__(self)
        new_node.parents=[node_a,node_b]
        new_node.name="BroadcastTo({})".format(node_a.name)
        return new_node
    def compute(self,node,vals):
        return np.broadcast_to(vals[0], shape=vals[1].shape)
    def gradient(self,node,grad):
        grad_A = reduce_sum(grad,new_axis=0)
        grad_B = zeros_like(node.parents[1])
        return [grad_A,grad_B]

class Conv2d(OP):
    def __call__(self, node_a,node_b,padding=(0, 0), stride=(1, 1)):
        new_node=OP.__call__(self)
        new_node.padding = padding
        new_node.stride = stride
        new_node.shape=[1,1,1,1]
        new_node.parents=[node_a,node_b]
        new_node.name="conv(node_a,node_b)"
        return new_node

    def compute(self, node, vals):
        t = make_padding(vals[0], node.padding)
        B, C, iH, iW = t.shape
        iC, oC, kH, kW = vals[1].shape
        assert C == iC, 'Conv2d channels in not equal.'
        out = batch_conv2d_f(t, vals[1], node.stride)
        
        return out
    def gradient(self, node, grad):
        gradA=conv2d_grad_x(grad, node.parents[1], node.padding,node.stride)
        gradB=conv2d_grad_bias(grad, node.parents[0], node.padding,node.stride)
        return [gradA,gradB]

class Conv2d_GradientXOp(OP):
    def __call__(self, node_a,node_b,padding=(0, 0), stride=(1, 1)):
        new_node=OP.__call__(self)
        new_node.padding = padding
        new_node.stride = stride
        print(new_node.stride,"xx")
        
        new_node.parents=[node_a,node_b]
        new_node.name="convgradx(node_a,node_b)"
        return new_node
    def compute(self,node,vals):

        return unwrap_padding(
            batch_conv2d_im_backward_f(vals[0], vals[1], node.stride),
            node.padding
        )
    def gradient(self,node,grad):
        raise "no grad"
class Conv2d_Gradient_BiasOp(OP):
    def __call__(self, node_a,node_b,padding=(0, 0), stride=(1, 1)):
        new_node=OP.__call__(self)
        new_node.padding = padding
        new_node.stride = stride
        
        new_node.parents=[node_a,node_b]
        new_node.name="convgradfilter(node_a,node_b)"
        return new_node
    def compute(self,node,vals):
        print(vals[0].shape,vals[1].shape)
        return batch_conv2d_weight_backward_f(
            vals[0],
            make_padding(vals[1], node.padding),
            node.stride
        )
    def gradient(self,node,grad):
        raise "no grad"


class MaxPool(OP):
    def __init__(self):
        self.res=np.ones(1)
        self.max_idx=np.ones(1)
        
    def __call__(self, node_a,ksize,padding=(0, 0), stride=(1, 1)):
        new_node=OP.__call__(self)
        new_node.ksize=ksize
        new_node.padding = padding
        new_node.stride = stride
        new_node.max_idx = np.ones(1)
        new_node.res = np.ones(1)
        

        new_node.parents=[node_a]
        global n

        new_node.name="pool(node_a)"+str(n)
        n=n+1
        return new_node

    def compute(self, node, vals):
        print(vals[0])
        
        vals[0] = make_padding(vals[0], node.padding)
        
        B, C, iH, iW = vals[0].shape
        print(vals[0].shape)
        global dx_shape
        dx_shape[node.name] = vals[0].shape
        vals[0]=vals[0].reshape(B*C,1,iH,iW)
        res = im2bchwkl(vals[0], ksize=node.ksize,stride=node.stride)
        res = res.reshape(int(B * C * iH* iW/(node.ksize[0]*node.ksize[1])),int(node.ksize[0]*node.ksize[1]))
        res = res.transpose(1,0)
        

        node.res = res
        self.res =res
        global gres
        gres = res 
        print(gres.shape,"dygres")
        
        # print(self.res)
        node.dx_shape=vals[0].shape
        node.max_idx = np.argmax(res, axis=0)
        self.max_idx = node.max_idx
        global gidx
        gidx [node.name]= node.max_idx
        # print(node.max_idx.shape,res.shape,node.max_idx.size,range(node.max_idx.size))
        
        out = res[node.max_idx, range(node.max_idx.size)]
        # print(out.shape)
        # exit()
        H = (iH-(node.ksize[0]-1)+1)/(node.stride[0])+1
        W = (iW-(node.ksize[1]-1)+1)/(node.stride[1])+1
        H =int(H)-1
        W = int(W)-1
        out = out.reshape(B,C,H,W)
        node.val = out
        
        return out
    def gradient(self, node, grad):
        return [maxpool_grad(node, grad,stride=node.stride,node_name=node.name)]

class Maxpool_GradientOp(OP):
    
    def __call__(self, node_a,node_b,padding=(0, 0), stride=(1, 1),node_name="pool"):
        new_node=OP.__call__(self)
        new_node.padding = padding
        new_node.stride = stride
        new_node.ksize= node_a.ksize
        new_node.parents=[node_a,node_b]
        new_node.node_name=node_name
        new_node.name="poolgradient(node_a)"
        return new_node
    def compute(self,node,vals):

        global gres
        global gidx
        global dx_shape
        global dX_col
        
        dX_col = np.zeros_like(gres)
        # vals[0].dtype="float32"
        dout_flat = vals[0].ravel()
        dX_col = np.zeros((dX_col.shape[0],dout_flat.shape[0]))
        print(gres.shape,dout_flat.shape,vals[0].shape,dX_col.shape,gidx[node.node_name].shape)
        
        dX_col[gidx[node.node_name],range(gidx[node.node_name].size)] = dout_flat
        dX_col = dX_col.T.reshape(dx_shape[node.node_name])

        dX_col = im2bchwkl(dX_col, ksize=(node.ksize[0],node.ksize[1]),\
            stride=(node.stride[0], node.stride[1]))
        # print(dout_flat)
        # print( vals[1])
        
        return unwrap_padding(
            dX_col.reshape(dx_shape[node.node_name]),
            node.padding
        )
    def gradient(self,node,grad):
        raise "no grad"

def batch_conv2d_f(x, kernel, stride=(1, 1)):
    x = im2bchwkl(x, kernel.shape[-2:], stride)
    print(x.shape,kernel.shape)
    return np.tensordot(x, kernel, [(1, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)


def batch_conv2d_weight_backward_f(kernel, input, stride=(1, 1)):
    '''kernel is result tensor grad, input is original tensor'''
    B, C, H, W = kernel.shape
    
    x = im2bchwkl(input, kernel.shape[-2:], stride = stride,dilation=stride)
    print(x.shape,kernel.shape)

    return np.tensordot(x, kernel, [(0, 4, 5), (0, 2, 3)]).transpose(0, 3, 1, 2)


def batch_conv2d_im_backward_f(x, kernel, stride=(1, 1)):
    '''input is result tensor grad, kernel is weight tensor'''
    ksize = kernel.shape
    x = dilate_input(x, stride)
    x = make_padding(x, ((ksize[2]-1), (ksize[3]-1)))
    return batch_transposed_conv2d_f(x, kernel, invert=True)


def batch_transposed_conv2d_f(x, kernel, invert=False):
    ksize = kernel.shape
    x = transpose_kernel(
        im2bchwkl(x, ksize[-2:])
    )
    i = 1 if invert else 0
    return np.tensordot(x, kernel, [(1, 4, 5), (i, 2, 3)]).transpose(0, 3, 1, 2)


def im2bchwkl(input, ksize, stride=(1, 1), padding=(0, 0), dilation=(1, 1), writeable=False):
    if padding != (0, 0):
        assert not writeable, 'No writable in padding mode.'
        input = make_padding(input, (padding[0], padding[1]))

    isize = input.shape
    istrides = input.strides

    H = (isize[2]-(dilation[0]*(ksize[0]-1)+1))/(stride[0])+1
    W = (isize[3]-(dilation[1]*(ksize[1]-1)+1))/(stride[1])+1
    assert int(H) == H and int(W) == W, 'conv2d not aligned'
    H = int(H)
    W = int(W)
    istrides = list(istrides+istrides[-2:])
    istrides[2] *= stride[0]
    istrides[3] *= stride[1]
    istrides[4] *= dilation[0]
    istrides[5] *= dilation[1]
    
    return np.lib.stride_tricks.as_strided(input,
                                           (isize[0], isize[1], H,
                                            W, ksize[0], ksize[1]),
                                           istrides,
                                           writeable=writeable,
                                           )


def make_padding(input, padding):
    if padding == (0, 0):
        return input
    b, c, h, w = input.shape
    p, q = padding
    result = np.zeros((b, c, h+2*p, w+2*q), dtype=np.float32)
    result[:, :, p:-p, q:-q] = input
    return result


def unwrap_padding(input, padding):
    if padding == (0, 0):
        return input
    p, q = padding
    return input[..., p:-p, q:-q]


def transpose_kernel(kernel):
    return kernel[..., ::-1, ::-1]


def dilate_input(input, stride=(1, 1)):
    if stride == (1, 1):
        return input
    isize = input.shape
    x = np.zeros((isize[0], isize[1], (isize[2]-1) *
                  stride[0]+1, (isize[3]-1)*stride[1]+1), dtype=np.float32)
    x[..., ::stride[0], ::stride[1]] = input
    return x

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
conv2d=Conv2d()
conv2d_grad_x=Conv2d_GradientXOp()
conv2d_grad_bias=Conv2d_Gradient_BiasOp()
maxpool = MaxPool()
maxpool_grad = Maxpool_GradientOp()
embed = EmbedOp()
embed_grad = EmbedGradientOp()