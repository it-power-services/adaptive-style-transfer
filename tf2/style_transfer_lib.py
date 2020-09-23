import random
import numpy as np
import tensorflow as tf

DSR = 0.8  # args.discr_success_rate


class options:
    is_training = True
    save_freq = 1000
    discr_loss_weight = 1
    transformer_loss_weight = 100
    feature_loss_weight = 100
    dsr = DSR
    lr = 0.0002
    gf_dim = 32
    df_dim = 64
    total_steps = int(3e5)  # 5021 #
    image_size = 768
    batch_size = 1
    path_to_content_dataset = "../../data/new_models/data_large/"
    path_to_art_dataset = "../../data/paul-cezanne/"
    path_to_models = "./models"
    style_name = "cezanne"

    disc_scale_weight = {
        "scale_0": 1.0,
        "scale_1": 1.0,
        "scale_3": 1.0,
        "scale_5": 1.0,
        "scale_6": 1.0,
    }


def read_image(path_to_image):
    tf.print('Reading image', path_to_image)
    image_file = tf.io.read_file(path_to_image)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    # convert to float32
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def get_batch(dataset, augmentor, batch_size=1):
    """
    Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
     also containing artist name, technique name, and period of creation for given artist.
     In case of content images we have only the 'path' column.
    Args:
        dataset: a list of paths to images (or style paintings)
        batch_size: size of batch
    Returns:
        dictionary with fields: image
    """
    # might be usefull here: https://www.tensorflow.org/api_docs/python/tf/TensorArray
    batch_image = []
    # --
    for i in range(batch_size):
        image = read_image(random.choice(dataset))
        #    height, width, channels = image.shape
        #    if min(height, width) <= options.image_size :
        #        scale_ratio = (options.image_size) / min(height, width)
        #        size = tf.cast([height * scale_ratio, width * scale_ratio], tf.int32)
        #        image = tf.image.resize(image, size=size,
        #            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        #     if min(height, width) <= options.image_size :
        #         tf.print('image too small')
        #         scale_ratio = (options.image_size*1.3) / min(height, width)
        #         size = tf.cast([height * scale_ratio, width * scale_ratio], tf.int32)
        #         image = tf.image.resize(image, size=size,
        #             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        #     if max(height, width) > options.image_size*3:
        #         scale_ratio = options.image_size / max(height, width)
        #         size = tf.cast([height * scale_ratio, width * scale_ratio], tf.int32)
        #         image = tf.image.resize(image, size=size,
        #             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
        batch_image.append(augmentor(image))
    return tf.convert_to_tensor(batch_image)


def batch_to_one_image(images_batch):
    b_size, im_h, im_w, im_c = images_batch.shape.as_list()
    images_batch_one = tf.reshape(images_batch, [b_size * im_h, im_w, im_c])
    return images_batch_one


def instance_norm(input, name="instance_norm"):
    depth = input.get_shape()[3]
    rn_init = tf.random_normal_initializer(mean=1.0, stddev=0.02)
    const_init = tf.constant_initializer(0.0)
    scale = tf.Variable(rn_init(shape=[depth], dtype=tf.dtypes.float32), name="scale")
    offset = tf.Variable(const_init(shape=[depth]), name="offset")
    mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (input - mean) * inv
    return scale * normalized + offset


def deconv2d(in_tensor, output_dim, ks=4, s=2, stddev=0.02, init=None, name="deconv2d"):
    # Upsampling procedure, like suggested in this article:
    # https://distill.pub/2016/deconv-checkerboard/. At first upsample
    # tensor like an image and then apply convolutions.
    in_tensor = tf.image.resize(
        images=in_tensor,
        size=tf.shape(in_tensor)[1:3] * s,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )  # That is optional
    cov_out = tf.keras.layers.Conv2D(
        filters=output_dim,
        kernel_size=ks,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=init,
        use_bias=False,
        name=name,
    )(in_tensor)
    return cov_out


def sce_criterion(logits, labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    )


def abs_criterion(pred, target):
    return tf.reduce_mean(tf.abs(pred - target))


def mse_criterion(pred, target):
    return tf.reduce_mean((pred - target) ** 2)


def bin_loss(discr_preds, truth_fun):
    """
    discr_preds: predictions returned by th discriminator (usually a dictionary of five tensors)
    truth_fun: sould be tf.onelike, or tf.zero_like (the function to use to get the target/label for the predicted tensor) 

    """
    return {
        key: sce_criterion(pred, truth_fun(pred)) * options.disc_scale_weight[key]
        for key, pred in discr_preds.items()
    }


## -----------------------------------------
## discriminator loss
## -----------------------------------------
def discriminator_loss(
    input_painting_discr_predictions,
    input_image_discr_predictions,
    output_image_discr_predictions,
):
    # The discriminator should know that the painting is real art --> we expect the prediction to be ones
    input_painting_discr_loss = bin_loss(input_painting_discr_predictions, tf.ones_like)
    # The discriminator should know that the input and output images are not real art --> we expect the prediction to be zeros
    input_image_discr_loss = bin_loss(input_image_discr_predictions, tf.zeros_like)
    output_image_discr_loss = bin_loss(output_image_discr_predictions, tf.zeros_like)
    return (
        tf.add_n(list(input_painting_discr_loss.values()))
        + tf.add_n(list(input_image_discr_loss.values()))
        + tf.add_n(list(output_image_discr_loss.values()))
    )


## -----------------------------------------
## generator loss
## -----------------------------------------
def generator_loss(
    output_discr_preds,
    input_image,
    output_image,
    output_image_features,
    input_image_features,
    options,
):
    # we want the discriminator to think the generated image is real art --> we expect the prediction to be ones
    loss = bin_loss(output_discr_preds, tf.ones_like)
    gener_loss = tf.add_n(list(loss.values()))
    # --
    output_image_pooled = tf.keras.layers.AvgPool2D(
        pool_size=10, strides=1, padding="same"
    )(output_image)
    input_image_pooled = tf.keras.layers.AvgPool2D(
        pool_size=10, strides=1, padding="same"
    )(input_image)
    img_loss = mse_criterion(output_image_pooled, input_image_pooled)
    # --
    feature_loss = abs_criterion(output_image_features, input_image_features)
    # --
    gen_total_loss = (
        options.discr_loss_weight * gener_loss
        + options.transformer_loss_weight * img_loss
        + options.feature_loss_weight * feature_loss
    )
    return gen_total_loss


## =================================================================================================================================================
## discriminator
## =================================================================================================================================================
def Discriminator(name="discriminator", input_shape=[256, 256, 3]):
    """
    Discriminator agent, that provides us with information about image plausibility at
    different scales.
    """
    # --
    turn_norm_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    inp = tf.keras.layers.Input(shape=input_shape, name="input_image_discr")
    # --
    h0_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 2,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h0_conv",
    )(inp)
    h0_norm = instance_norm(h0_cov, name="d_bn0")
    h0 = tf.keras.layers.LeakyReLU(alpha=0.2)(h0_norm)
    h0_pred = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=5,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h0_pred",
    )(h0)
    # --
    h1_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 2,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h1_conv",
    )(h0)
    h1_norm = instance_norm(h1_cov, name="d_bn1")
    h1 = tf.keras.layers.LeakyReLU(alpha=0.2)(h1_norm)
    h1_pred = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=10,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h1_pred",
    )(h1)
    # --
    h2_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 4,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h2_conv",
    )(h1)
    h2_norm = instance_norm(h2_cov, name="d_bn2")
    h2 = tf.keras.layers.LeakyReLU(alpha=0.2)(h2_norm)
    # --
    h3_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 8,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h3_conv",
    )(h2)
    h3_norm = instance_norm(h3_cov, name="d_bn3")
    h3 = tf.keras.layers.LeakyReLU(alpha=0.2)(h3_norm)
    h3_pred = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=10,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h3_pred",
    )(h3)
    # --
    h4_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 8,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h4_conv",
    )(h3)
    h4_norm = instance_norm(h4_cov, name="d_bn4")
    h4 = tf.keras.layers.LeakyReLU(alpha=0.2)(h4_norm)
    # --
    h5_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 16,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h5_conv",
    )(h4)
    h5_norm = instance_norm(h5_cov, name="d_bn5")
    h5 = tf.keras.layers.LeakyReLU(alpha=0.2)(h5_norm)
    h5_pred = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=6,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h5_pred",
    )(h5)
    # --
    h6_cov = tf.keras.layers.Conv2D(
        filters=options.df_dim * 16,
        kernel_size=5,
        strides=2,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h6_conv",
    )(h5)
    h6_norm = instance_norm(h6_cov, name="d_bn6")
    h6 = tf.keras.layers.LeakyReLU(alpha=0.2)(h6_norm)
    h6_pred = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="d_h6_pred",
    )(h6)
    # --
    return tf.keras.Model(
        inputs=[inp],
        outputs={
            "scale_0": h0_pred,
            "scale_1": h1_pred,
            "scale_3": h3_pred,
            "scale_5": h5_pred,
            "scale_6": h6_pred,
        },
        name=name,
    )


## =================================================================================================================================================


## =================================================================================================================================================
## generator
## =================================================================================================================================================
## ===============
## encoder
## ===============
def encoder(name="encoder", input_shape=[256, 256, 3]):
    turn_norm_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    inp = tf.keras.layers.Input(shape=input_shape, name="input_image_encoder")
    image = instance_norm(inp, name="g_e0_bn")
    c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
    # --
    c1_cov = tf.keras.layers.Conv2D(
        filters=options.gf_dim,
        kernel_size=3,
        strides=1,
        padding="valid",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_e1_c",
    )(c0)
    c1_norm = instance_norm(c1_cov, name="g_e1_bn")
    c1 = tf.keras.layers.ReLU()(c1_norm)
    # --
    c2_cov = tf.keras.layers.Conv2D(
        filters=options.gf_dim,
        kernel_size=3,
        strides=2,
        padding="valid",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_e2_c",
    )(c1)
    c2_norm = instance_norm(c2_cov, name="g_e2_bn")
    c2 = tf.keras.layers.ReLU()(c2_norm)
    # --
    c3_cov = tf.keras.layers.Conv2D(
        filters=options.gf_dim * 2,
        kernel_size=3,
        strides=2,
        padding="valid",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_e3_c",
    )(c2)
    c3_norm = instance_norm(c3_cov, name="g_e3_bn")
    c3 = tf.keras.layers.ReLU()(c3_norm)
    # --
    c4_cov = tf.keras.layers.Conv2D(
        filters=options.gf_dim * 4,
        kernel_size=3,
        strides=2,
        padding="valid",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_e4_c",
    )(c3)
    c4_norm = instance_norm(c4_cov, name="g_e4_bn")
    c4 = tf.keras.layers.ReLU()(c4_norm)
    # --
    c5_cov = tf.keras.layers.Conv2D(
        filters=options.gf_dim * 8,
        kernel_size=3,
        strides=2,
        padding="valid",
        activation=None,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_e5_c",
    )(c4)
    c5_norm = instance_norm(c5_cov, name="g_e5_bn")
    c5 = tf.keras.layers.ReLU()(c5_norm)
    # --
    return tf.keras.Model(inputs=[inp], outputs=c5, name=name)


## ===============
## decoder
## ===============
def decoder(name="decoder", input_shape=[16, 16, 256]):
    """
    the decoder takes the shape of the image features tensor as input
    i.e., the shape of the tensor outputed by the encoder (encoder().output.shape[1:4])
    """
    turn_norm_init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
    inp = tf.keras.layers.Input(shape=input_shape, name="input_image_feat")
    # --
    def residule_block(x, dim, ks=3, s=1, name="res"):
        p = int((ks - 1) / 2)
        y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
        y = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=ks,
            strides=s,
            padding="valid",
            activation=None,
            kernel_initializer=turn_norm_init,
            use_bias=False,
            name=name + "_c1",
        )(y)
        y = instance_norm(y, name=name + "_bn1")
        y = tf.pad(
            tf.keras.layers.ReLU()(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT"
        )
        y = tf.keras.layers.Conv2D(
            filters=dim,
            kernel_size=ks,
            strides=s,
            padding="valid",
            activation=None,
            kernel_initializer=turn_norm_init,
            use_bias=False,
            name=name + "_c2",
        )(y)
        y = instance_norm(y, name=name + "_bn2")
        return y + x

    # --
    # Now stack 9 residual blocks
    num_kernels = inp.get_shape().as_list()[-1]
    r1 = residule_block(inp, num_kernels, name="g_r1")
    r2 = residule_block(r1, num_kernels, name="g_r2")
    r3 = residule_block(r2, num_kernels, name="g_r3")
    r4 = residule_block(r3, num_kernels, name="g_r4")
    r5 = residule_block(r4, num_kernels, name="g_r5")
    r6 = residule_block(r5, num_kernels, name="g_r6")
    r7 = residule_block(r6, num_kernels, name="g_r7")
    r8 = residule_block(r7, num_kernels, name="g_r8")
    r9 = residule_block(r8, num_kernels, name="g_r9")
    # --
    # Decode image.
    d1 = deconv2d(r9, options.gf_dim * 8, 3, 2, init=turn_norm_init, name="g_d1_dc")
    d1 = tf.keras.layers.ReLU()(instance_norm(input=d1, name="g_d1_bn"))
    d2 = deconv2d(d1, options.gf_dim * 4, 3, 2, init=turn_norm_init, name="g_d2_dc")
    d2 = tf.keras.layers.ReLU()(instance_norm(input=d2, name="g_d2_bn"))
    d3 = deconv2d(d2, options.gf_dim * 2, 3, 2, init=turn_norm_init, name="g_d3_dc")
    d3 = tf.keras.layers.ReLU()(instance_norm(input=d3, name="g_d3_bn"))
    d4 = deconv2d(d3, options.gf_dim, 3, 2, init=turn_norm_init, name="g_d4_dc")
    d4 = tf.keras.layers.ReLU()(instance_norm(input=d4, name="g_d4_bn"))
    d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
    # --
    # pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c'))*2. - 1.
    d4_cov = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=7,
        strides=1,
        padding="valid",
        activation=tf.keras.activations.sigmoid,
        kernel_initializer=turn_norm_init,
        use_bias=False,
        name="g_pred_c",
    )(d4)
    pred = tf.subtract(tf.multiply(d4_cov, tf.constant(2.0)), tf.constant(1.0))
    return tf.keras.Model(inputs=[inp], outputs=pred, name=name)


## ===============
## generator
## ===============
def Generator(name="generator", input_shape=[256, 256, 3]):
    input_image = tf.keras.layers.Input(shape=input_shape, name="input_image_gener")
    input_features = encoder(name="encoder_input", input_shape=input_shape)(input_image)
    output_image = decoder(input_shape=input_features.shape[1:4])(input_features)
    output_features = encoder(name="encoder_output", input_shape=input_shape)(
        output_image
    )
    return tf.keras.Model(
        inputs=[input_image],
        outputs={
            "input_features": input_features,
            "output_image": output_image,
            "output_features": output_features,
        },
        name=name,
    )
