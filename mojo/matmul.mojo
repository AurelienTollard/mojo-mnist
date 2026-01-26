# Tiled implementation of matmul. Should be good enough for this project
fn matmul_gpu_kernel[
    dtype: DType,
    layout_a: Layout,
    layout_b: Layout,
    layout_c: Layout,
](
    a: LayoutTensor[dtype, layout_a, ImmutAnyOrigin],
    b: LayoutTensor[dtype, layout_b, ImmutAnyOrigin],
    c: LayoutTensor[dtype, layout_c, MutAnyOrigin],
):
    pass
