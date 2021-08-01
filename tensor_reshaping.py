import torch 

x = torch.arange(9)

y = x.reshape(3,3)

print(y)

print(y.t())
z = y.t()

#================================================#
#           Reshape vs view                      #
#================================================#

# not countigues
print(z.contiguous().view(9))
# print(z.view(9))


print(z.reshape(9))


k = torch.arange(0,17,2).reshape(3,3)
print('k')
print(k)
print(k.view(9))

#================================================#
#            Concat vs stack                     #
#================================================#

a = torch.rand((4,7))
b = torch.rand((4,7))
print('concat')
print(torch.cat((a,b), dim=0).shape)
print(torch.cat((a,b), dim=1).shape)
print('stack')
print(torch.stack((a,b)).shape)
print(torch.stack((a,b), dim=0).shape)
print(torch.stack((a,b), dim=1).shape)

#================================================#
#            Unravel                             #
#================================================#


r = torch.rand((4,5))
print('reshape -1')
print(r.reshape(-1))
print(r.reshape(2, -1))


#================================================#
#            transpose vs permute                #
#================================================#

#transpose is special case of permute
d = torch.arange(48).reshape(2,4,6)
print(d.transpose(1,2).shape)
print(d.permute(0,2,1).shape)

#================================================#
#           unsqueeze                            #
#================================================#
o = torch.rand((4,5))
print('unsqueeze')
print(o.unsqueeze(0).shape)
print(o.unsqueeze(1).shape)
print(o.unsqueeze(2).shape)
# print(o.unsqueeze(2))