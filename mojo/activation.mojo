@always_inline
fn relu[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Relu Op using the equation $max(x, 0)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the RELU operation on.

    Returns:
        The result of the RELU operation.
    """
    return max(x, 0)

def main():
    pass
