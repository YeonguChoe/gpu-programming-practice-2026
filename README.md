# GPU Programming Practice

## Grid Dimesion



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

<image src="matrix-multiplication.png">

```python
@triton.jit
def matrix_multiplication(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K: tl.constexpr,  # Because K is used by tl.arange
    N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    block_m_indice = pid_0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n_indice = pid_1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_row_indice = tl.arange(0, K)

    # Thinking in point
    # A(m, k): a_ptr + m * K + k
    a_ptrs = a_ptr + block_m_indice[:, None] * K + b_row_indice[None, :]

    # Thinking in point
    # B(k, n): b_ptr + k * N + n
    b_ptrs = b_ptr + b_row_indice[:, None] * N + block_n_indice[None, :]

    a_mask = block_m_indice[:, None] < M
    b_mask = block_n_indice[None, :] < N
    c_mask = (block_m_indice[:, None] < M) & (block_n_indice[None, :] < N)

    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    c = tl.dot(a, b)

    # Thinking in point
    # C(m, n): output_ptr + m * N + n
    c_ptrs = c_ptr + block_m_indice[:, None] * N + block_n_indice[None, :]
    tl.store(c_ptrs, c, mask=c_mask)
```

