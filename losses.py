import tensorflow as tf

def gram_matrix(input_tensor):
    """
    Returns the gram matrix of the input tensor
    """
    #The below uses einstein summation, which I hadn't seen before.
    #Basically, take the two tensors, with indices bij and then c or d as the last one.
    #Then, we sum the products of the elements at each index listed.
    #In this case, for each of the last two channels (c and d), sum over their product for all b, i, and js, and place those into a new matrix.
    return tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

#Gatys' content loss: half the sum of the MSEs between the output and the target
def gatys_content_loss(output_dict, target_dict):
    return tf.add_n([tf.reduce_sum(tf.square(output_dict[layer] - target_dict[layer])) for layer in target_dict.keys()]) / 2

#Gatys' style loss: the sum of the MSEs between the output and target gram matrices, which is then divided by four times the number of feature maps squared times the number of pixels in the layer squared
#Expects layer name: gram matrix dictionaries for the first two, and a matching layer name: (original network layer) tensor shape dictionary in NHWC form for the third argument
def gatys_style_loss(output_dict, target_dict, dim_dict):
    divisor_dict = {name: 4 * (dims[3] ** 2) * ((dims[1] * dims[2]) ** 2) for name, dims in dim_dict.items()}
    return tf.add_n([tf.reduce_sum(tf.square(output_dict[layer] - target_dict[layer])) / divisor_dict[layer] for layer in target_dict.keys()])

def shifted_gram_matrix(input_tensor, delta):
    """
    Returns a gram-like matrix formed when each index is multiplied by the one delta indices in both x and y dimensions.
    https://arxiv.org/pdf/1606.01286.pdf - Berger and Memisevic
    """
   #Shift in the column direction, get adjusted gram matrices, then average
    mat_one = input_tensor[:,:,:-delta,:]
    mat_two = input_tensor[:,:,delta:,:]

    result_one = tf.linalg.einsum('bijc,bijd->bcd', mat_one, mat_two)

    rows, cols = mat_one.shape[1], mat_one.shape[2]
    matrix_size = tf.cast(rows * cols, tf.float32)

    #Repeat with the row direction
    mat_one = input_tensor[:,:-delta,:,:]
    mat_two = input_tensor[:,delta:,:,:]

    result_two = tf.linalg.einsum('bijc,bijd->bcd', mat_one, mat_two)

    rows, cols = mat_one.shape[1], mat_one.shape[2]
    matrix_size = tf.cast(rows * cols, tf.float32)

    return (result_one, result_two)

def shifted_gram_loss(output_dict, target_dict, dim_dict):
    return tf.add_n([shifted_gram_cost(output_dict[name], target_dict[name]) for name in output_dict.keys()]) / 2

def shifted_gram_cost(outputs, targets):
    generated_x, generated_y = outputs
    target_x, target_y = targets
    loss = tf.norm(generated_x - target_x, ord = 'euclidean') ** 2 + tf.norm(generated_y - target_y, ord = 'euclidean') ** 2
    loss /= 2
    return loss

