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

```python
@triton.jit
def matrix_vector_mutiply(matrix_ptr, vector_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    row_pid = tl.program_id(0)
    columns_vector = tl.arange(0, BLOCK_SIZE)
    matrix_offsets = row_pid * BLOCK_SIZE + columns_vector
    vector_offsets = columns_vector
    mask = columns_vector < BLOCK_SIZE
    row_from_matrix = tl.load(matrix_ptr + matrix_offsets, mask=mask)
    column_from_vector = tl.load(vector_ptr + vector_offsets, mask=mask)
    multiplied_vector = row_from_matrix * column_from_vector
    sum = tl.sum(multiplied_vector)
    tl.store(output_ptr + row_pid, sum)
```

## Matrix Multiplication

```python

```


## Matrix Transpose

## Dot Product

## Cross Product