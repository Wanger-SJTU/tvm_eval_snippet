
from turtle import pd
import tvm
import numpy as np
from tvm import te, relay
from tvm import topi


data = relay.var("data", shape=(1, 360, 784, 1), dtype="float32")
weight = relay.var("weight", shape=(8, 8, 1, 1), dtype="float32")

res = relay.nn.conv2d_transpose(data, weight, strides=(1,1), padding=(1,1,1,1), data_layout="NHWC", kernel_layout="HWIO")
mod = tvm.IRModule.from_expr(res)

lib = relay.build(mod, target="llvm")
dev = tvm.cpu()
import pdb
pdb.set_trace()
res = lib.executor.benchmark(lib.module.entry_name, dev)()
print(res.median)