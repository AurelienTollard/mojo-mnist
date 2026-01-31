@always_inline
fn relu_scalar[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    """Compute ReLU: max(x, 0)."""
    return max(x, Scalar[dtype](0))


@always_inline
fn identity_scalar[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return x
