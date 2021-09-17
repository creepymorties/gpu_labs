import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler
import numpy as np
import math
import sys
import timeit
from PIL import Image



def image_blur(input_image: str, output_image: str):

    # ################################################# #
    # load image in to array and extract color channels #
    # ################################################# #
    try:
        img = Image.open(input_image)
        input_array = np.array(img)
        red_channel = input_array[:, :, 0].copy()
        green_channel = input_array[:, :, 1].copy()
        blue_channel = input_array[:, :, 2].copy()
    except FileNotFoundError:
        sys.exit("Cannot load image file")


    # ######################################## #
    # generate gaussian kernel (size of N * N) #
    # ######################################## #
    sigma = 2  # standard deviation of the distribution
    kernel_width = int(3 * sigma)
    if kernel_width % 2 == 0:
        kernel_width = kernel_width - 1  # make sure kernel width only sth 3,5,7 etc

    # create empty matrix for the gaussian kernel #
    kernel_matrix = np.empty((kernel_width, kernel_width), np.float32)
    kernel_half_width = kernel_width // 2
    for i in range(-kernel_half_width, kernel_half_width + 1):
        for j in range(-kernel_half_width, kernel_half_width + 1):
            kernel_matrix[i + kernel_half_width][j + kernel_half_width] = (
                    np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
                    / (2 * np.pi * sigma ** 2)
            )
    gaussian_kernel = kernel_matrix / kernel_matrix.sum()


    # #################################################################### #
    # calculate the CUDA threads/blocks/gird base on width/height of image
    # #################################################################### #
    height, width = input_array.shape[:2]
    dim_block = 32
    dim_grid_x = math.ceil(width / dim_block)
    dim_grid_y = math.ceil(height / dim_block)

    # load CUDA code

    kernel = '''
   __global__ void applyFilter(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {

    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row < height && col < width) {
        const int half = kernelWidth / 2;
        float blur = 0.0;
        for(int i = -half; i <= half; i++) {
            for(int j = -half; j <= half; j++) {

                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));

                const float w = kernel[(j + half) + (i + half) * kernelWidth];
                blur += w * input[x + y * width];
            }
        }
        output[col + row * width] = static_cast<unsigned char>(blur);
    }
} 
   '''


    # mod = compiler.SourceModule(open('gaussian_blur.cu').read())
    mod = compiler.SourceModule(kernel)
    apply_filter = mod.get_function('applyFilter')

    # ##################
    # apply the  filter
    # ##################
    # start time
    time_started = timeit.default_timer()
    for channel in (red_channel, green_channel, blue_channel):
        apply_filter(
            drv.In(channel),
            drv.Out(channel),
            np.uint32(width),
            np.uint32(height),
            drv.In(gaussian_kernel),
            np.uint32(kernel_width),
            block=(dim_block, dim_block, 1),
            grid=(dim_grid_x, dim_grid_y)
        )
    # end time
    time_ended = timeit.default_timer()


    # ####################################################################### #
    # create the output array with the same shape and type as the input array #
    # ####################################################################### #
    output_array = np.empty_like(input_array)
    output_array[:, :, 0] = red_channel
    output_array[:, :, 1] = green_channel
    output_array[:, :, 2] = blue_channel

    # save result image
    Image.fromarray(output_array).save(output_image)

    # display total time
    print('Total processing time: ', time_ended - time_started, 's')


image_blur(input_image='./resources/blur/to_blur.jpg', output_image='./resources/blur/blured.jpg')