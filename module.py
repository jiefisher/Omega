import node
class Module:
    def parameters(self):
        params = [getattr(self, i) for i in dir(self) if isinstance(
            getattr(self, i), node.Parameter)]
        submodules = [getattr(self, i) for i in dir(
            self) if isinstance(getattr(self, i), Module)]
        for p in params:
            yield p
        for sm in submodules:
            yield from sm.parameters()

    def __call__(self, *args):
        return self.forward(*args)
