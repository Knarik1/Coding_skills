import torch 
import numpy as np

#================================================#
#           Tensor indexing                      #
#================================================#

a = torch.arange(7,20)
print(a)
get_idx = [1, 6, 8]
print(a[get_idx])


# new
print(a[[1,6]])


b = torch.rand((3,4))
print('b')
print(b)

print(b[[1,2], [1,3]])
print(b[1:2, 1:3])


print(b[b>0.7])
print(np.where(b>0.7,1, 0))

m = torch.tensor([1,1,3,4,5,3])
print('unique')
print(m.unique())