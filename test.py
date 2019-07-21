import torch.nn as nn
import torch
from torch.autograd import Variable

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)
x = Variable(torch.rand(2,3,1,1))
print(x.size())
y = conv1x1(3, 3)(x)
print(y.size())

# z = torch.cat(x, x)
# print(z.size())

t = torch.mul(x, x)
print(t.size())

