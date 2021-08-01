import torch 
import numpy as np

#================================================#
#                  Init tensors                  #
#================================================#
my_tensor = torch.Tensor()
print(my_tensor) 

arr = [[1,2,3], [4,5,6]]
my_tensor = torch.tensor(arr)
print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.requires_grad)
print(my_tensor.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_tensor = torch.tensor(arr, dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor)


#================================================#
#            Other methods of init               #
#================================================#
my_tensor = torch.zeros((2,3))
print('zeros')
print(my_tensor)

my_tensor = torch.ones((2,3))
print('ones')
print(my_tensor)


my_tensor = torch.empty((2,3))
print('empty')
print(my_tensor)


my_tensor = torch.eye(3)
print('eye')
print(my_tensor)

my_tensor = torch.abs(torch.eye(3) - torch.ones((3,3)))
print('without diag')
print(my_tensor)


my_tensor = torch.zeros((2,3)) + 1
print('not square ones')
print(my_tensor)

arr2 =  np.array([[1,2], [3,4]])
my_tensor2 = torch.tensor(arr2)
arr2[0][0] = 9
print('arr2')
print(arr2)
print(my_tensor2)



my_tensor3 = torch.clone(my_tensor2)
my_tensor2[0][0] = 7
print("tensor3")
print(my_tensor3)

# uniform
my_tensor = torch.rand((2,3))
print("rand")
print(my_tensor)

print(torch.eye(5, 3))

my_tensor = torch.arange(1,5)
print('arage')
print(my_tensor)

my_tensor = torch.linspace(1,7,3)
print("linespace")
print(my_tensor)


# new 
my_tensor = torch.empty((2,3)).normal_(mean=0, std=3)
print('normal')
print(my_tensor)
print('mean', np.mean(my_tensor.numpy()), 'std', np.std(my_tensor.numpy()))


my_tensor = torch.empty((2,3)).uniform_(2,5)
print("uniform")
print(my_tensor)


my_tensor = torch.diag(torch.ones(3))
print('diag')
print(my_tensor)


#================================================#
#            Convert tensors                     #
#================================================#
my_tensor = torch.arange(4)
print('convertions')
print(my_tensor.bool())
print(my_tensor.long())
print(my_tensor.short())
print(my_tensor.float())


#================================================#
#            From numpy by link                  #
#================================================#
arr3 = np.array([[1,1,2], [3,3,4]])
my_tensor = torch.from_numpy(arr3)
print('numpy arr')
arr3[0][0] = 9


print(my_tensor)
print(my_tensor.numpy())


#================================================#
#            Matrix multiplication               #
#================================================#
a = torch.tensor([[1,2,3], [4,5,6]])
b = torch.tensor([[1,1], [5,5], [8,8]])

print('multi')
print(torch.mm(a,b))
print(a@b)

c = torch.rand((3,4,6))
d = torch.rand((3,6,2))

print('batch')
print(torch.bmm(c,d).shape)



f = torch.rand((8,3,4,6)).reshape(8*3, 4, 6)
k = torch.rand((8,3,6,2)).reshape(8*3, 6, 2)
print(f.shape)

print('batch 4 dim ??????')
print(torch.bmm(f,k).shape)


print('exponent')
print(torch.exp(a))
print(torch.pow(a, 4))


#================================================#
#            Matrix ops                          #
#================================================#
my_tensor = my_tensor.type(torch.float32)
my_tensor = my_tensor.float()
print("----------------")
print(my_tensor)
print('---------------')
print('sum', torch.sum(my_tensor))
print('sum', torch.sum(my_tensor, dim=0))
print('mean', torch.mean(my_tensor, dim=1))
print('argmin', torch.argmin(my_tensor, dim=0))
print('argmax', torch.argmax(my_tensor, dim=1))
print("=================")
print('max', torch.max(my_tensor, dim=1))
print('max', torch.max(my_tensor, dim=0))

print('sort')
sorted_vals, sorted_idx = torch.sort(my_tensor, axis=1)
print(sorted_vals)

# wow == ReLU
z = torch.clamp(sorted_vals, min=3)
print('clamp')
print(z)


q = torch.tensor([1,2,0,2,1,9])
print('any')
print(torch.any(q))
print('all')
print(torch.all(q))