import op
class Node:
    def __init__(self,name='',parents=[]):
        self.name=name
        self.op=None
        self.parents=[parents] if type(parents)==Node else parents
    
    def __add__(self,other):
        if type(other)==Node:
            return op.add_op(self,other)
        else:
            return op.add_by_const(self,other)
    def __sub__(self,other):
        if type(other)==Node:
            return op.sub_op(self,other)
        else:
            return op.sub_by_const_op(self,other)
    def __rsub__(self,other):
        if type(other)!=Node:
            return op.const_by_sub_op(self,other)
    def __neg__(self):
        return op.neg_op(self)    
    def __mul__(self,other):
        if type(other)==Node:
            return op.mul_op(self,other)
        else:
            return op.mul_by_const(self,other)
    def __div__(self,other):
        if type(other)==Node:
            return op.div_op(self,other)
        else:
            return op.div_by_const_op(self,other)
    __radd__ = __add__
    __rmul__ = __mul__
    __rdiv__ = __div__


def Variable(name):
    placeholder_node = op.place_holder()
    placeholder_node.name = name
    return placeholder_node

def Parameter(name,const):
    parameter_node = op.place_holder()
    parameter_node.name = name
    parameter_node.const = const
    return parameter_node
# a=Node("a")
# b=Node("b")
# c=a+b
# print(c.op)