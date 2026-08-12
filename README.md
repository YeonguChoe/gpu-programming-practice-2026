# GPU Programming Practice

## Add two 1D vectors

```python
@triton.jit
def add(x_ptr, y_ptr, out_ptr, total_data, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    sum = x + y
    tl.store(out_ptr + offsets, sum, mask=mask)
```

## Reversing 1D vector

```python
@triton.jit
def reverse(read_ptr, write_ptr, total_data, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    read_offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    write_offsets = (total_data - 1) - read_offsets
    mask_on_current_offsets = read_offsets < total_data
    data = tl.load(read_ptr + read_offsets, mask=mask_on_current_offsets)
    tl.store(write_ptr + write_offsets, data, mask=mask_on_current_offsets)
```

## Matrix Vector Multiplication


## Matrix Multiplication


## Matrix Transpose

## Dot Product

## Cross Product