from functools import partial
import argparse

import numpy as np

from utils import msgranul
from utils.inout import InOutLoop
from utils.imgproc import read_img, write_img, copy_border, remove_border
from utils.csv import save_maxlocals

def read_img_mask_border(filename, padding=0, invert=False):
    input = read_img(filename, filename_mask=filename.replace('inputs', 'outputs_unet'))
    if padding > 0:
        input[0] = copy_border(input[0], padding)
        input[1] = copy_border(input[1], padding)
    if invert:
        input[1] = 255 - input[1]
    return input

def generate_kernels(type='c', sides=[160, 160], octave=16, min_size=0.15):
    kernels = msgranul.createkernels(type, sides[0], sides[1], octave, min_size)
    print ( "\n kernels: " + str( len(kernels)))
    return kernels

def correlate_image_and_kernels(img, kernels):
    locals = msgranul.correlate(img, kernels)
    print ( "\n maxlocals before apply granulometry: " + str( len(locals)))
    return locals

def apply_granulometry_based_correlation(img, locals, min_corr=0.2, max_inter=0.4):
    locals = msgranul.apply(img, locals, min_corr,  max_inter)[0]
    print ( "\n maxlocals after apply granulometry: " + str( len(locals)))
    return locals

def output_img(img, locals):
    return msgranul.printmaxlocals(img, locals)

def granul(input, type='c', sides=[160, 160], octave=16,  min_corr=0.2, max_inter=0.4, min_size=0.15):
    target_image = input[1]
    kernels = generate_kernels(type, sides, octave, min_size)

    maxlocals = correlate_image_and_kernels(target_image, kernels)
    maxlocals = apply_granulometry_based_correlation(target_image, maxlocals, min_corr, max_inter)
    
    return [output_img(input[0], maxlocals), maxlocals]

def remove_padding(maxlocals, padding):
    max_locals_no_padding = []
    for idx, maxlocal in enumerate(maxlocals):
        maxlocal.x -= padding
        maxlocal.y -= padding

        max_locals_no_padding.append(maxlocal)

    return max_locals_no_padding


def write_img_border(filename_img, output, padding=0):
    out_img = output[0]
    maxlocals = output[1]

    if padding > 0:
        out_img = remove_border(out_img, padding)
        maxlocals = remove_padding(maxlocals, padding)

    maxlocals_filename = filename_img.replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
    save_maxlocals(maxlocals, maxlocals_filename)

    write_img(filename_img, out_img)

def run_granul(inputs, outputs, side=[60, 60], min_size=0.15, padding=24, octave=8, max_inter=0.1, min_corr=0.52):

    mainloop = InOutLoop(input_folder=inputs, output_folder=outputs,
                         extensions=['jpeg', 'png', 'jpg'])

    mainloop.on_input(partial(read_img_mask_border, padding=padding, invert=True))
    mainloop.on_run(partial(granul, type='c', sides=tuple(side), 
                                    octave=octave,  min_corr=min_corr, 
                                    max_inter=max_inter, min_size=min_size))
    mainloop.on_output(partial(write_img_border, padding=padding))

    mainloop.run()