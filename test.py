import numpy as np
from op import *
import op
import gradients
import node
import executor
import module
import linear
import loss
import opt
class mynet(module.Module):
    def __init__(self,shapes):
        self.shapes = shapes
        self.fc=linear.Linear(self.shapes)
    def forward(self,x):
        y=self.fc(x)
        return y

labels=node.Node("label")
c = mynet(shapes=(4,2))
x = node.Node("x")
loss = loss.cross_entropy(c(x),labels)
a=[2*np.ones(1*4).reshape(1,4),np.ones(1*4).reshape(1,4)]
b=[np.array([1,0]).reshape(1,2),np.array([0,1]).reshape(1,2)]
for epoch in range(10):
    for batch in range(2):
        optimizer=opt.SGD(loss,c.parameters())
        
        optimizer.step(feed_dict={x:a[batch],labels:b[batch]})
print(optimizer.parameters[1].const,optimizer.parameters[0].const)

def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = (m-BSZ[0])
    ncols = (n-BSZ[1])
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]
a=np.arange(4*4).reshape(4,4)
print(a)
print(im2col_sliding_strided(a,(2,2),stepsize=1).shape)

def im2bchwkl(input, ksize, stride=(1, 1), padding=(0, 0), dilation=(1, 1), writeable=False):
    if padding != (0, 0):
        assert not writeable, 'No writable in padding mode.'
        input = make_padding(input, (padding[0], padding[1]))
    print(input)
    isize = input.shape
    istrides = input.strides

    H = (isize[2]-(dilation[0]*(ksize[0]-1)+1))/(stride[0])+1
    W = (isize[3]-(dilation[1]*(ksize[1]-1)+1))/(stride[1])+1
    assert int(H) == H and int(W) == W, 'conv2d not aligned'
    H = int(H)
    W = int(W)
    print(H,W)
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
def unwrap_padding(input, padding):
    if padding == (0, 0):
        return input
    p, q = padding
    return input[..., p:-p, q:-q]
a=np.arange(3*1*4*4).reshape(3,1,4,4)
res = im2bchwkl(a, ksize=(2,2),stride=(2, 2))
# print(res.shape)
res = res.reshape(12,4)
ms =res
res = res.transpose(1,0)
print(res)
max_idx = np.argmax(res, axis=0)
print(max_idx)

out = res[max_idx, range(max_idx.size)]
print(out)
# Reshape to the output size: 14x14x5x10
out = out.reshape(3,1,2,2)
print(out)

dX_col = np.zeros_like(res)
# flatten the gradient
dout_flat = out.ravel()
print(dX_col.shape,dout_flat.shape,max_idx.shape)
dX_col[max_idx,range(max_idx.size)] = dout_flat
print(dX_col)
print(dX_col.shape)
dX_col = dX_col
dX_col = im2bchwkl(dX_col, ksize=(2,2),stride=(1, 1))
print(dX_col.shape)
print(dX_col.reshape(3,1,4,4).transpose(0, 3, 1, 2))
print(a)
# get the original X_reshaped structure from col2im

# dX = col2im_indices(dX_col,a.shape,self.size,self.size,padding=0,stride=self.stride)
# dX = dX.reshape(self.n_X,self.d_X,self.h_X,self.w_X)

# a=np.ones(1*1*4*4).reshape(1,1,4,4)
# b=np.ones(1*1*2*2).reshape(1,1,2,2)


# x=node.Node("x")
# y=node.Node("y")
# z=conv2d(x,y)
# grads = gradients.gradients(z,[x,y])
# exe=executor.Executor([z]+grads)
# output = exe.run(feed_dict={x:a,y:b})
# print(output[2].shape)
# print(output[1].shape)
# print(output[0].shape)