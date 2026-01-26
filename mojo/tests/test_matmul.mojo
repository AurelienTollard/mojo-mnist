from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from layout.tensor_core import TensorCore
from layout.layout_tensor import copy_dram_to_sram_async
from utils import Index
from sys import size_of, argv
from testing import assert_equal, assert_almost_equal


fn main():
    with DeviceContext() as ctx:
        out_tensor = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out_tensor.enqueue_fill(0)
        input_tensor1 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        input_tensor2 = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)

        with input_tensor1.map_to_host() as input_tensor1_host, input_tensor2.map_to_host() as input_tensor2_host:
            for row in range(SIZE):
                for col in range(SIZE):
                    val = row * SIZE + col
                    input_tensor1_host[row * SIZE + col] = val
                    input_tensor2_host[row * SIZE + col] = Float32(2.0) * val

            for i in range(SIZE):
                for j in range(SIZE):
                    for k in range(SIZE):
                        expected[i * SIZE + j] += (
                            input_tensor1_host[i * SIZE + k] * input_tensor2_host[k * SIZE + j]
                        )

            with out_tensor.map_to_host() as out_tensor:
                for i in range(SIZE * SIZE):
                    assert_almost_equal(out_tensor[i], expected[i])

            print("Matmul passed")
