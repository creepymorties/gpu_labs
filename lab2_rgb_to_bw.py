from PIL import Image
import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy


def cuda_rgb_to_bw(inPath, outPath):
    totalT0 = time.clock()

    im = Image.open(inPath)
    px = numpy.array(im)
    px = px.astype(numpy.float32)

    getAndConvertT1 = time.clock()

    allocT0 = time.clock()
    d_px = cuda.mem_alloc(px.nbytes)
    cuda.memcpy_htod(d_px, px)

    allocT1 = time.clock()

    # Kernel declaration
    kernelT0 = time.clock()

    # Kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024, 1, 1)
    checkSize = numpy.int32(im.size[0] * im.size[1])
    grid = (int(im.size[0] * im.size[1] / BLOCK_SIZE) + 1, 1, 1)

    # Kernel text
    kernel = """

    __global__ void bw( float *inIm, int check ){

        int idx = (threadIdx.x ) + blockDim.x * blockIdx.x ;
        if(idx *3 < check*3)
        {
       		int val = 0.21 *inIm[idx*3] + 0.71*inIm[idx*3+1] + 0.07 * inIm[idx*3+2];
        	inIm[idx*3]= val;
        	inIm[idx*3+1]= val;
        	inIm[idx*3+2]= val;
        }
    }
    """

    # Compile and get kernel function
    mod = SourceModule(kernel)
    func = mod.get_function("bw")
    func(d_px, checkSize, block=block, grid=grid)

    kernelT1 = time.clock()

    # Get back data from gpu
    backDataT0 = time.clock()

    bwPx = numpy.empty_like(px)
    cuda.memcpy_dtoh(bwPx, d_px)
    bwPx = (numpy.uint8(bwPx))

    backDataT1 = time.clock()

    # Save image
    storeImageT0 = time.clock()
    pil_im = Image.fromarray(bwPx, mode="RGB")

    pil_im.save(outPath)

    totalT1 = time.clock()

    getAndConvertTime = getAndConvertT1 - totalT0
    allocTime = allocT1 - allocT0
    kernelTime = kernelT1 - kernelT0
    backDataTime = backDataT1 - backDataT0
    storeImageTime = totalT1 - storeImageT0
    totalTime = totalT1 - totalT0

    print("Black and white image")
    print("Image size: ", im.size)
    print("Time taken to get and convert image data: ", getAndConvertTime)
    print("Time taken to allocate memory on the GPU: ", allocTime)
    print("Kernel execution time: ", kernelTime)
    print("Time taken to get image data from GPU and convert it: ", backDataTime)
    print("Time taken to save the image: ", storeImageTime)
    print("Total execution time : ", totalTime)


cuda_rgb_to_bw(inPath='./resources/rgb_to_bw/to_convert.jpg', outPath='./resources/rgb_to_bw/converted.jpg')