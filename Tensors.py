# This file is for learning `Learn The Basic` Pytorch tutorials
# This part learn about Tensor.

import torch 
import numpy as np

# Directly from data 
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

# From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From another tensor
x_ones = torch.ones_like(x_data) # retains the properties(shape, datatype) of x_data
print(f"Ones Tensor :  \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor : \n {x_rand} \n")

# With random or constant values 
shape = (2,3,)  # <- shape is a tuple of tensor dimensions.
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# TO CHEKE avalability of CUDA
print(torch.cuda.is_available())
tensor = torch.rand(shape)
if torch.cuda.is_available():
    tensor = tensor.to('cuda') # <- tensor can move to cuda memory by using 'tensor.to('cuda')'

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
tensor = torch.ones(4, 4)
tensor[:,1] = 0
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print('==============')
print(y1)
print('==============')

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print('==============')
print(z1)
print('==============')

print('this is check for usage of item()')
print()
agg = tensor.sum()
print(agg)
print()
agg_item = agg.item()
print(agg_item, type(agg_item))