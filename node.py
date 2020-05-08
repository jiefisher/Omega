import op
class Node:
    def __init__(self,name='',parents=[]):
        self.name=name
        self.op=None
        self.parents=[parents] if type(parents)==Node else parents
    
    def __add__(self,other):
        return op.add_op(self,other)
    
    def __mul__(self,other):
        return op.mul_op(self,other)


def Variable(name):
    placeholder_node = op.place_holder()
    placeholder_node.name = name
    return placeholder_node


# a=Node("a")
# b=Node("b")
# c=a+b
# print(c.op)