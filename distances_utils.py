import keras.backend as K

def euclidean_distance(vects):

    x, y = vects

    x = K.l2_normalize(x)
    y = K.l2_normalize(y)

    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    #return K.sqrt(sum_square)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

## Cosine Loss
def cosine_loss(target, left, right):
    normalized_left = tf.nn.l2_normalize(left,0)
    normalized_right = tf.nn.l2_normalize(right,0)
    cosine = tf.reduce_sum(tf.multiply(normalized_left,normalized_right))

    return tf.reduce_sum(tf.pow(target-cosine, 2))

## Contrastive Loss
def contrastive_loss(y_true, y_pred):

    #y_pred = K.print_tensor(y_pred, message='y_pred = ')

    margin=K.constant(0.5)
    #epsilon=K.constant(1e-6)

    #y_pred = y_pred+epsilon

    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))

    # keep the similar label (1) close to each other
    similarity = 0.5 * (1.0 - y_true) * square_pred

    # give penalty to dissimilar label if the distance is bigger than margin
    dissimilarity = 0.5 * y_true * margin_square

    return K.mean(dissimilarity + similarity) #/ 2

    #return K.mean(y_true * square_pred + (1.0 - y_true) * margin_square)
