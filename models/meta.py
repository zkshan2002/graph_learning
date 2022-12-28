import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class MetaModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        return

    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, named_grads, lr):
        for (name, value), grad in zip(self.named_params(self), named_grads):
            new_value = value - grad * lr
            self.set_param(self, name, new_value)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', autograd.Variable(ignore.weight.data, requires_grad=True))
        if ignore.bias is not None:
            self.register_buffer('bias', autograd.Variable(ignore.bias.data, requires_grad=True))
        else:
            self.bias = None
        return

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        result = [('weight', self.weight)]
        if self.bias is not None:
            result += [('bias', self.bias)]

        return result
