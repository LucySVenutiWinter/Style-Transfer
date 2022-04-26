import tensorflow as tf
import numpy as np
from functools import partial

# Loads the file and converts it to three-channel float tensor of the right dimensionality, where each value is between 0 and 1 (or 0 and 2^<size> if uint)
def load_image(filepath, size = None, img_dtype = tf.dtypes.uint8):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_image(image, channels = 3)
        image = tf.image.convert_image_dtype(image, img_dtype)#tf.float32)#img_dtype)

        if (size):
            image = tf.image.resize(image, size)

        image = image[tf.newaxis, :]# Adds an extra axis, as the network will expect the first index to be the example number
      
        #image *= 255

        return image

#Below functions are for luminance style transfer.
#1) use separate_luminance to get RGB versions of the luminance channel only of the content and style images
#2) run transfer
#3) then use recombine_yiq_to_rgb with the iq channels of the content from separate_luminance

#Separate luminance transforms an rgb image into a yiq image.
#Returns (y, iq) if lum_as_rgb is false, otherwise returns (rgb replication of y, iq).
def separate_luminance(input_tensor, lum_as_rgb=True):
    yiq_rep = tf.image.rgb_to_yiq(input_tensor)
    y_channel = tf.expand_dims(yiq_rep[:,:,:,0], axis = -1)
    iq_channel = yiq_rep[:,:,:,1:]

    if lum_as_rgb:
       image = luminance_only_to_rgb(y_channel)
       return (image, iq_channel) 
    return (y_channel, iq_channel)

def luminance_only_to_rgb(luminance):
    luminance = tf.concat([luminance, tf.zeros_like(luminance), tf.zeros_like(luminance)], axis = -1)
    return tf.image.yiq_to_rgb(luminance)

#Reverses separate_luminance.
def recombine_yiq_to_rgb(luminance, iq_channel, lum_as_rgb=True):
    if lum_as_rgb:
        luminance = tf.expand_dims(tf.image.rgb_to_yiq(luminance)[:,:,:,0], axis = -1)
    image = tf.concat([luminance, iq_channel], axis = -1)

    return tf.image.yiq_to_rgb(image)

#Takes luminance images and matches mean/variance so they are similar.
def match_luminance(match_to, match_from):
    mean_to, var_to = tf.nn.moments(match_to, axes = [1,2])
    mean_from, var_from = tf.nn.moments(match_from, axes = [1,2])
    return var_to / var_from * (match_from - mean_from) + mean_to

#The following functions take a content and style image and return a three-element tuple.
#The first element is the RGB version of the content image to run,
#The second element is the RGB version of the style image to run, 
#and the third elemnt is a function to execute on the final image which will restore the content's color profile
def prep_luminance_transfer(content_image, style_image):
    content, content_iq = separate_luminance(content_image)
    style, _ = separate_luminance(style_image)
    return (content,
            style,
            partial(recombine_yiq_to_rgb, iq_channel=content_iq, lum_as_rgb=True))

#As above, but matches the luminance profiles between the images first.
def prep_histogram_matched_luminance_transfer(content_image, style_image):
    content_y, content_iq = separate_luminance(content_image, False)
    style_y, _ = separate_luminance(style_image, False)
    style_y = match_luminance(content_y, style_y)
    return (luminance_only_to_rgb(content_y),
            luminance_only_to_rgb(style_y),
            partial(recombine_yiq_to_rgb, iq_channel=content_iq, lum_as_rgb=True))
