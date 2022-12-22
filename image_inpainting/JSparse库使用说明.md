# JSparse计算库介绍
10月20日，计图团队发布了稀疏卷积计算库JSparse，可支持稀疏卷积，稀疏矩阵乘法等。
## JSparse安装

1. 克隆`https://github.com/Jittor/JSparse.git`
2. 两种方法：
   1. 复制出里面的文件夹jsparse到你的运行目录下。
   2. 执行`python setup.py install`安装JSparse

## 使用方法
下面以稀疏矩阵乘向量的代码为例展示计算库使用方法。

```python
import numpy as np
import jittor as jt
import jsparse.nn.functional as F
jt.flags.nvcc_flags += '-lcusparse'
jt.flags.use_cuda = 1

size = 1024

### construct sparse matrix
mat = np.random.rand(size,size).astype(np.float32)
mat = np.where(mat > 0.995, mat, 0)
indices = np.nonzero(mat)
values = mat[indices]

### construct vector
vec = np.random.rand(size,1).astype(np.float32)

## execute sparse matrix multiply a vector
output = F.spmm(
        rows=jt.array(indices[0]), 
        cols=jt.array(indices[1]), 
        vals=jt.array(values), 
        size=(size,size), 
        mat=jt.array(vec), 
        cuda_spmm_alg=1)

print(output)
```