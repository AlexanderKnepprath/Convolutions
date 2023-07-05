import random
import time

from scipy.signal import fftconvolve
import tensorflow as tf
import numpy

# A program which performs 2D Convolutions
pad_constant = 0


def standard_convolution(image, kernel):
    """Performs a convolution using standard techniques"""
    flipped_kernel = flip_kernel(kernel)  # rotates kernel about each axis
    kernel_size_x = len(flipped_kernel[0])
    kernel_size_y = len(flipped_kernel)
    output_image = []
    for i in range(1 - kernel_size_y, len(image)):
        output_row = []
        for j in range(1 - kernel_size_x, len(image[0])):
            sum = convolve(image, flipped_kernel, j, i)
            output_row.append(sum)
        output_image.append(output_row)
    return output_image


def convolve(image, kernel, x, y):
    """Performs a convolution with the flipped kernel placed over the image with (x,y) corresponding to the
    top left coordinate of the kernel. x and y may be negative, but should not exceed the bounds of the image."""
    sum = 0
    for i in range(len(kernel)):
        for j in range(len(kernel[i])):
            # implicit padding: if the index would be out of bounds, then pad.
            if y+i < 0 or y+i >= len(image) or x+j < 0 or x+j >= len(image[0]):
                sum += kernel[i][j] * pad_constant
            else:
                sum += image[y+i][x+j] * kernel[i][j]
    return sum


def flip_kernel(kernel):
    """Flips the kernel on both axes"""
    new_kernel = []
    for i in range(len(kernel)):
        new_kernel_row = []
        for j in range(len(kernel[i])):
            # appends the diagonally symmetrical value to the new kernel array
            new_kernel_row.append(kernel[len(kernel)-i-1][len(kernel[i])-j-1])
        new_kernel.append(new_kernel_row)
    return new_kernel


def imp_fourier_convolution(matrix, kernel):
    return fftconvolve(matrix, kernel)


def fourier_convolution(matrix, kernel):
    x_size = tf.shape(matrix)[0]
    y_size = tf.shape(matrix)[1]
    matrix_transform = tf.signal.rfft2d(matrix, fft_length=[x_size, y_size])
    kernel_transform = tf.signal.rfft2d(kernel, fft_length=[x_size, y_size])  # sizes will automatically 0-pad kernel
    multiplication = matrix_transform * kernel_transform
    convolution = tf.signal.irfft2d(multiplication)
    return convolution


def print_image(image):
    for i in image:
        print(i)


def run_benchmarking(runs_per, max_pow, grow_kernel, convolution_type):
    for i in range(max_pow + 1):
        # print("....")
        total_time = 0
        for j in range(runs_per):
            # print("...")
            image = []
            kernel = []
            for k in range(2 ** (i + 1)):
                # print("..")
                image_row = []
                for l in range(2 ** (i + 1)):
                    # print(".")
                    image_row.append(numpy.round(random.random()*100))
                image.append(image_row)

            if grow_kernel:
                for k in range(2 ** (i + 1)):
                    kernel_row = []
                    for l in range(2 ** (i + 1)):
                        kernel_row.append(numpy.round(random.random()*100))
                    kernel.append(kernel_row)
            else:
                for k in range(2):
                    kernel_row = []
                    for l in range(2):
                        kernel_row.append(numpy.round(random.random()*100))
                    kernel.append(kernel_row)

            if convolution_type == 0:
                init_time = time.perf_counter()
                standard_convolution(image, kernel)
                total_time += time.perf_counter() - init_time
            else:
                init_time = time.perf_counter()
                imp_fourier_convolution(image, kernel)
                total_time += time.perf_counter() - init_time

        avg_time = total_time / runs_per
        type_str = ""
        if convolution_type == 0:
            type_str = "Standard Convolution: "
        else:
            type_str = "Fourier Convolution: "
        print(type_str + str(2 ** i) + "x" + str(2 ** i) + " image, grow_kernel = " + str(grow_kernel) + " | Avg. Time: " + str(avg_time))


if __name__ == '__main__':
    image = [[1, 2, 3, 2, 1],
             [2, 4, 7, 4, 2],
             [3, 7, 10, 7, 3],
             [2, 4, 7, 4, 2],
             [1, 2, 3, 2, 1]]
    kernel = [[0,0,1,0,0],
              [0,0,0,0,0],
              [5,0,0,0,5],
              [0,0,0,0,0],
              [0,0,1,0,0]]
    # print_image(standard_convolution(image, kernel))
    # print("---")
    # print_image(imp_fourier_convolution(image, kernel))
    # print("---")
    # print(fourier_convolution(image, kernel))

    #run_benchmarking(3, 5, False, 0)
    print("--")
    #run_benchmarking(3, 5, True, 0)
    print("--")
    run_benchmarking(3, 5, False, 1)
    print("--")
    run_benchmarking(3, 5, True, 1)
