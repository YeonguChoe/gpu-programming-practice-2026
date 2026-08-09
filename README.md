# GPU Programming Practice

## Add two 1D vectors

```python
@triton.jit
def add(x_ptr, y_ptr, out_ptr, element_count, BLOCK_SIZE: tl.constexpr):
    # setup
    pid=tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    predicate = offsets < element_count

    # calculation
    x = tl.load(x_ptr+offsets,mask=predicate)
    y = tl.load(y_ptr+offsets,mask=predicate)
    result=x+y
    tl.store(out_ptr+offsets,result,mask=predicate)
```

