import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
from pycuda.compiler import SourceModule
from scipy import signal as sg
from scipy import misc


# DEVICE SETUP

BLOCK_SIZE = 32  # Max 32. 32**2 = 1024, max for GTX1060

kernel = '''
__global__ void conv(const float *A, const float *B, int aw, int ah, int bw, int bh, int b_sum, float *C){

    /*Get row and column to operate on from thread coordinates*/
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;
    
    /*Calculate "padding" radius of convolution kernel (distance around central pixel)*/
    int pw = (bw-1)/2;
    int ph = (bh-1)/2;

    /*If within the range of C (ie A - padding)*/
    if( row < (ah-2*ph) && col < (aw-2*pw) ) {
        
        /*Set initial pixel value*/
        int val = 0;
        
         /*For each vertical position on the kernel matrix, relative to the central pixel*/
        for(int i=-ph; i<=ph; i=i+1){
            /*Calculate zero-indexed row ID on kernel matrix*/
            int b_row = i+ph; 

            /*For each horizontal position on the kernel matrix, relative to the central pixel*/
            for(int j=-pw; j<=pw; j=j+1){
                /*Calculate zero-indexed column ID on kernel matrix*/
                int b_col = j+pw;

                /*Add product of kernel value and corresponding image value to running total*/
                val += A[ (row+ph +i)*aw + (col+pw +j) ] * B[ b_row*bw + b_col ];
            }
        }
        
        /*Copy appropriately normalised resulting pixel value to position on C matrix*/
        C[row*(aw-2*pw) + col] = val/b_sum;
    }
}

'''


# Compile kernel
mod = SourceModule(kernel)

# Get functions
conv = mod.get_function("conv")


def convolve(a, b):
    global BLOCK_SIZE
    global conv

    a, b = [np.array(i).astype(np.float32) for i in [a, b]]

    # Matrix A
    aw = np.int32(a.shape[1])  # Widthof in matrix
    ah = np.int32(a.shape[0])  # Height of in matrix

    # Matrix B (kernel)
    bw = np.int32(b.shape[1])  # Widthof in matrix
    if bw % 2 == 0:
        print("Kernel width is not an odd number! Strange things will happen...")
    bh = np.int32(b.shape[0])  # Height of in matrix
    if bh % 2 == 0:
        print("Kernel height is not an odd number! Strange things will happen...")
    b_sum = np.int32(np.absolute(b).sum())

    # Matrix C, subtract 2*padding, *2 because it's taken off all sides
    c = np.empty([ah - (bh - 1), aw - (bw - 1)])
    c = c.astype(np.float32)

    # Allocate memory on device
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Copy matrix to memory
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Set grid size from A matrix
    grid = (int(aw / BLOCK_SIZE + (0 if aw % BLOCK_SIZE is 0 else 1)),
            int(ah / BLOCK_SIZE + (0 if ah % BLOCK_SIZE is 0 else 1)),
            1)

    # Call gpu function
    conv(a_gpu, b_gpu, aw, ah, bw, bh, b_sum, c_gpu, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=grid)

    # Copy back the result
    cuda.memcpy_dtoh(c, c_gpu)

    # Free memory. May not be useful? Ask about this.
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()

    # Return the result
    return c


k_sv = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]


a = Image.open('.\\resources\\blur\\to_blur.jpg')

px = np.array(a)
px = px.astype(np.float32)

c = convolve(px, k_sv)

pil_im = Image.fromarray(c, mode="RGB")
pil_im.save('.\\resources\\convol\\convol.jpg')