import cupy

x_gpu = cupy.ones((32, 768), dtype='int8')
(hist, bin_edges) = cupy.histogram(x_gpu)


