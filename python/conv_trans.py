
import tvm
import numpy as np
from tvm import te
from tvm import topi


Input = te.placeholder(shape=[1, 1, 480, 784])
Filter = te.placeholder(shape=[1, 1, 8, 8])
strides = [2, 2]
padding = [1,1,1,1] 
out_dtype = "float32"
output_padding = [1,1]
out = topi.nn.conv2d_transpose_nchw(Input, Filter, strides, padding, out_dtype, output_padding)

sch = te.create_schedule(out.op)

print(tvm.lower(sch, [Input, Filter, out]))

def arr(x):
    shape = [item.value for item in x.shape]
    return tvm.nd.array(np.random.random(shape).astype("float32"))

data = map(arr, [Input, Filter, out])

lib = tvm.build(sch, [Input, Filter, out])
dev = tvm.cpu()
res = lib.time_evaluator(lib.entry_name, dev)(*data)
print(res.median)