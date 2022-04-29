import tensorflow as tf
import image_utils
import losses
from functools import partial
import os

#This script defines a lot of variables up front in order to keep all the controls in one place.

#Where generated files go.
GENERATED_FILEPATH = "generated"

#Which layers are used to calculate loss.
CONTENT_LAYERS = ['block4_conv2']
STYLE_LAYERS = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
        ]

#What functions calculate loss - as well as what function to apply to style layers to capture style.
#Default style function is gram_matrix, and losses are gatys_style_loss and gatys_content_loss.
#See losses.py for function signatures.
STYLE_FUNCTION = partial(losses.gram_matrix)#Partial is necessary if there are hyperparameters to pass to the function
STYLE_LOSS = losses.gatys_style_loss
CONTENT_LOSS = losses.gatys_content_loss

OPTIMIZER = tf.optimizers.Nadam(0.5)#We want to take large steps
#Things appear to work fairly well with 1e-4, 2.5e-8, 1e-6
STYLE_WEIGHT = 1e-4
CONTENT_WEIGHT = 2.5e-8
TOTAL_VARIATION_WEIGHT = 1e-4

WEIGHTS = {'style': STYLE_WEIGHT, 'content': CONTENT_WEIGHT, 'variation': TOTAL_VARIATION_WEIGHT}

#What sizes to do the coarse to fine adjustments with
SIZES = [(600, 600), (900, 900)]

#Number of steps to do per algorithm iteration, and number of iterations to do.
#The script will output the transferred image at the end of each iteration.
NUM_ITERATIONS = 50
STEPS_PER_ITER = 100

#The images to use. (content, style, initial). If initial is None, then the algorithm will initialize the image with noise.
IMAGE_LIST = [
        ("data/pur.jpg", "data/leaves.jpg", "data/pur.jpg"),
        ("data/pur.jpg", "data/wave.jpg", "data/pur.jpg")
        ]

##########################################################################################

#These are not used, this is just a list of all possible layers so that it's easy to play around
layers = ['block1_conv1',
          'block1_conv2',
          'block1_pool',
          'block2_conv1',
          'block2_conv2',
          'block2_pool',
          'block3_conv1',
          'block3_conv2',
          'block3_conv3',
          'block3_conv4',
          'block3_pool',
          'block4_conv1',
          'block4_conv2',
          'block4_conv3',
          'block4_conv4',
          'block4_pool',
          'block5_conv1',
          'block5_conv2',
          'block5_conv3',
          'block5_conv4',
          'block5_pool']

##########################################################################################

def extract_path_tail(filepath):
    filepath = filepath.split("/")[-1]
    filepath = filepath.split(".")
    if len(filepath) == 1:
        return filepath
    filepath = ".".join(filepath[:-1])
    return filepath

#Converts vgg from maxpool to avgpool (though it should work on any model)
def convert_to_avgpool(vgg_model):
    to_replace = []
    for layer in range(len(vgg_model.layers)):
        if isinstance(vgg_model.layers[layer], tf.keras.layers.MaxPooling2D):
            to_replace.append(layer)
    for replacee in to_replace:
        layer = vgg_model.layers[replacee] 
        vgg_model.layers[replacee] = tf.keras.layers.AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                data_format=layer.data_format,
                name=layer.name)
    return vgg_model

def vgg_layers(layer_names):
    """
    Given a list of VGGnet layer names, returns a model that will output those layers
    """
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    model = convert_to_avgpool(model)

    return model

class StyleContentModel(tf.keras.models.Model):
    """
    When called on an image, returns two dictionaries containing the contents and gram matrices of the outputs.
    Note that preprocessing expects images to be in the 0, 255 range.
    """
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, style_function=STYLE_FUNCTION):
        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_dims = {style_name: mat.shape for style_name, mat in zip(self.style_layers, style_outputs)}

        style_outputs = [style_function(style_output) for style_output in style_outputs]

        contents_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': contents_dict, 'style': style_dict, 'dimensions': style_dims}

#Makes sure an image's pixel magnitudes are within [0, 255]
def clip_image(image):
    return tf.clip_by_value(image, 0.0, 255.0)

#Find and return the style and content losses for the given images, targets, and weights
def style_content_loss(outputs, targets, weights, style_loss_function=STYLE_LOSS, content_loss_function=CONTENT_LOSS):
    style_outputs = outputs['style']
    style_dims = outputs['dimensions']
    content_outputs = outputs['content']
    style_targets = targets['style']
    content_targets = targets['content']
    style_weight = weights['style']
    content_weight = weights['content']
    
    style_loss = style_loss_function(style_outputs, style_targets, style_dims)
    style_loss *= style_weight / len(style_outputs)
    content_loss = content_loss_function(content_outputs, content_targets)
    content_loss *= content_weight / len(content_outputs)

    return style_loss, content_loss

#Perform a single gradient step on the image
def train_step(image, model, targets, weights):
    with tf.GradientTape() as tape:
        outputs = model(image)
        style_loss, content_loss = style_content_loss(outputs, targets, weights)
        loss = style_loss + content_loss
        variation_loss = (weights['variation'] * tf.image.total_variation(image))
        loss += variation_loss 

    grad = tape.gradient(loss, image)

    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_image(image))

#Uses model to step image towards the given targets with the weights.
#This performs steps_per_iter descent steps on the image, then saves. It does this for num_iters iterations.
def gen_image(image, model, targets, weights, num_iters=100, steps_per_iter=100, save_dir="default", postprocess=None):
    print("Saving to " + save_dir)
    if (save_dir not in os.listdir(GENERATED_FILEPATH)):
        os.mkdir(GENERATED_FILEPATH + "/" + save_dir)
    for i in range(num_iters):
        style_loss, content_loss = style_content_loss(model(image), targets, weights)
        for j in range(steps_per_iter):
            train_step(image, model, targets, weights)
        if postprocess == None:
            to_save = image
        else:
            to_save = postprocess(image)
        tf.keras.preprocessing.image.save_img(f"{GENERATED_FILEPATH}/{save_dir}/{str(i+1)}.png", tf.squeeze(to_save))
        print("Finished iteration " + str(i + 1))

def coarse_to_fine(images, paths, weights, sizes, num_iters=100, steps_per_iter=100, lum=None):
    """
    Uses coarse-to-fine image generation from a content, style, and init image.
    images should be a tuple of content, style, and initial images.
    paths should be those images' pathnames.
    weights must be a dictionary containing "content," "style," and "variation" keys and float values.
    sizes should be a list of tuples containing the height and width of the image to be generated for each step.
    Smaller sizes should be given first, as these will be processed and used as seeds for later sizes.
    Note that the aspect ratio is preserved, so really you're setting a max bound on width/height, not dimension values.
    The style image will also be scaled down, but won't be scaled up if it's smaller. This may affect results.
    If lum is None, does normal process. If lum is "nomatch does luminance only transfer.
    If lum is "match," then does luminance only transfer with histogram matching.
    """
    for new_size in sizes:
        if new_size == sizes[0]:
            content_image, style_image, init_image = images
        else:
            content_image, style_image, _ = images
        print("Now generating images at size " + str(new_size))
        content = tf.image.resize(content_image, new_size, preserve_aspect_ratio=True)
        if tf.reduce_max(style_image.shape) > max(new_size):
            style = tf.image.resize(style_image, new_size, preserve_aspect_ratio=True)
        else:
            style = style_image
        init = tf.Variable(tf.image.resize(init_image, new_size, preserve_aspect_ratio=True))

        if lum == "nomatch":
            content, style, postprocess = image_utils.prep_luminance_transfer(content, style)
        elif lum == "match":
            content, style, postprocess = image_utils.prep_histogram_matched_luminance_transfer(content, style)
        elif lum == None:
            postprocess = None
        else:
            raise Exception("Given bad luminance transfer request")

        feature_extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)
        style_targets = feature_extractor(tf.constant(style))['style']
        content_targets = feature_extractor(tf.constant(content))['content']
        targets = {'style': style_targets, 'content': content_targets}

        #Files will be saved in folders bearing the name of the content and style images for easy recordkeeping.
        content_path, style_path, _ = paths
        save_dir = extract_path_tail(content_path) + "_CONTENT_" + extract_path_tail(style_path) + "_STYLE"
        if lum:
            save_dir += "_" + lum
        gen_image(init, feature_extractor, targets, weights, num_iters, steps_per_iter, save_dir, postprocess)

        init_image = init

def generate_coarse_to_fine(image_paths, weights, sizes, num_iters=100, steps_per_iter=100, lum=None):
    """
    Generates images using the coarse-to-fine technique. See coarse_to_fine for information on arguments.
    image_paths should be a list containing tuples of (content, style, initial) image filenames.
    If an initial image filename is None, the image will instead be initialized with noise.
    """
    for paths in image_paths:
        content_path, style_path, initial_path = paths

        content_image = image_utils.load_image(content_path)
        style_image = image_utils.load_image(style_path)
        if initial_path == None:
            initial_image = make_noise(content_image.shape)
        else:
            initial_image = image_utils.load_image(initial_path)

        images = (content_image, style_image, initial_image)
        coarse_to_fine(images, paths, weights, sizes, num_iters, steps_per_iter, lum)

def make_noise(shape=(224, 224)):
    return tf.random.uniform((1, shape[0], shape[1], 3), 0, 255)

#optimizer is used by the actual stepping function
optimizer = OPTIMIZER
generate_coarse_to_fine(IMAGE_LIST, WEIGHTS, SIZES, NUM_ITERATIONS, STEPS_PER_ITER)
generate_coarse_to_fine(IMAGE_LIST, WEIGHTS, SIZES, NUM_ITERATIONS, STEPS_PER_ITER, "nomatch")
generate_coarse_to_fine(IMAGE_LIST, WEIGHTS, SIZES, NUM_ITERATIONS, STEPS_PER_ITER, "match")
