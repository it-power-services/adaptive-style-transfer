import numpy as np
import tensorflow as tf
import cv2

# import tensorflow_addons as tfa


class Augmentor:
    def __init__(
        self,
        crop_size=(768, 768),
        scale_augm_prb=0.5,
        scale_augm_range=0.2,
        rotation_augm_prb=0.5,
        rotation_angle_range=20,
        hsv_augm_prb=1.0,
        hue_augm_range=13,
        saturation_augm_range=1.5,
        value_augm_range=0.5,
        affine_trnsfm_prb=0.5,
        #        affine_trnsfm_range=0.06,
        affine_stretch_range=0.2,
        affine_shear_factor=0.2,
        horizontal_flip_prb=0.5,
        vertical_flip_prb=0.5,
    ):

        self.crop_size = crop_size

        self.scale_augm_prb = scale_augm_prb
        self.scale_augm_range = scale_augm_range

        self.rotation_augm_prb = rotation_augm_prb
        self.rotation_angle_range = rotation_angle_range

        self.hsv_augm_prb = hsv_augm_prb
        self.hue_augm_range = hue_augm_range
        self.saturation_augm_range = saturation_augm_range
        self.value_augm_range = value_augm_range

        self.affine_trnsfm_prb = affine_trnsfm_prb
        #        self.affine_trnsfm_range = affine_trnsfm_range
        self.affine_stretch_range = affine_stretch_range
        self.affine_shear_factor = affine_shear_factor

        self.horizontal_flip_prb = horizontal_flip_prb
        self.vertical_flip_prb = vertical_flip_prb

    def __call__(self, image):

        ## apply the pipeline of augmentations.
        # random scale
        if self.scale_augm_prb > np.random.uniform():
            image = self.random_scale(image)

        # pad with reflection (for rotation and shear)
        height, width, ch = image.shape
        hgt_4 = height // 4
        wid_4 = width // 4
        paddings = [[hgt_4, hgt_4], [wid_4, wid_4], [0, 0]]
        image = tf.pad(image, mode="reflect", paddings=paddings)

        # random rotation
        if self.rotation_augm_prb > np.random.uniform():
            image = self.random_rotate(image)

        # random affine/shear
        if self.affine_trnsfm_prb > np.random.uniform():
            image = self.random_affine(image)

        # remove padding
        image = image[hgt_4:-hgt_4, wid_4:-wid_4, :]

        # random hsv transformation
        if self.hsv_augm_prb > np.random.uniform():
            try :
                image = self.random_hsv_transform(image)
            except :
                tf.print('[!] Could not apply an HSV transformation on the image')

        height, width, ch = image.shape
        try:
            # random crop of desired size.
            image = self.random_crop(image=image, crop_size=self.crop_size)
        except:
            # scale the image before cropping if the size is not big enough
            scale_ratio = self.crop_size[0] / min(height, width)
            size = tf.cast([height * scale_ratio, width * scale_ratio], tf.int32)
            image = tf.image.resize(
                image, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
            # take a random crop after scaling
            image = self.random_crop(image=image, crop_size=self.crop_size)

        # flip horizontally or vertically
        if self.horizontal_flip_prb > np.random.uniform():
            image = self.horizontal_flip(image)

        if self.vertical_flip_prb > np.random.uniform():
            image = self.vertical_flip(image)

        return image

    def random_scale(self, image):

        scale_h = 1 + np.random.uniform(
            low=-self.scale_augm_range, high=self.scale_augm_range
        )
        scale_w = 1 + np.random.uniform(
            low=-self.scale_augm_range, high=self.scale_augm_range
        )
        height, width, ch = image.shape
        size = tf.cast([height * scale_h, width * scale_w], tf.int32)
        image = tf.image.resize(
            image, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return image

    def random_crop(self, image, crop_size=(256, 256)):

        size = [crop_size[0], crop_size[1], 3]
        image = tf.image.random_crop(image, size)
        return image

    def horizontal_flip(self, image):

        return tf.image.flip_left_right(image)

    def vertical_flip(self, image):

        return tf.image.flip_up_down(image)

    def random_rotate(self, image):

        height, width, ch = image.shape
        angle = np.random.uniform(
            low=-self.rotation_angle_range, high=self.rotation_angle_range
        )
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        image = cv2.warpAffine(image.numpy(), M, (width, height))
        # image = imutils.rotate(image.numpy(), angle)
        return tf.convert_to_tensor(image)

    def random_hsv_transform(self, image):
        # To understaand HSV better:
        # * https://en.wikipedia.org/wiki/Hue
        # * https://en.wikipedia.org/wiki/Hue#/media/File:HSV_cone.jpg
        hue_delta = np.random.randint(-self.hue_augm_range, self.hue_augm_range,)
        saturation_mult = 1 + np.random.uniform(
            -self.saturation_augm_range, self.saturation_augm_range
        )
        value_mult = np.random.uniform(
            1 - self.value_augm_range, 1 + self.value_augm_range
        )

        image_hsv = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2HSV)
        image_hsv[:, :, 0] = image_hsv[:, :, 0] + hue_delta
        image_hsv[:, :, 1] *= saturation_mult
        image_hsv[:, :, 2] *= value_mult
        image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
        return tf.convert_to_tensor(image)

    def random_affine(self, image):
        """
        An affine transform, given (a, b, c, d, e, f), will move each pixel
        from position (x, y) to (a*x + b*y + c, d*x + e*y + f)

        intuitively (simplified):
        a, e : stretching/shrenking factors (negative numbers here return black images)
        b, d : shearing factors
        c, f : amount of shift
        """
        height, width, ch = image.shape
        # stretching
        a, e = np.random.uniform(
            1 - self.affine_stretch_range, 1 + self.affine_stretch_range, 2
        )
        # shearing
        b, d = np.random.uniform(-self.affine_shear_factor, self.affine_shear_factor, 2)
        # no shifting
        c, f = 0, 0
        transform_matrix = np.array([[a, b, c], [d, e, f]])
        # transform and keep the size
        image = cv2.warpAffine(image.numpy(), transform_matrix, (width, height))
        return tf.convert_to_tensor(image)

    def show_im(self, image, conversion=cv2.COLOR_RGB2BGR):
        """
        The function converts image to GBR and shows the image 
        in a pop-up screen, which closes by clicking any key in the keyboard

        conversion: takes any cv2 color flag. Available flags can be checked 
        by running 
        flags = [i for i in dir(cv2) if i.startswith('COLOR_')] 
        Default cv2.COLOR_RGB2BGR (under the assumption the input image is RGB)
        """
        cv2.imshow("image", cv2.cvtColor(image, conversion))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_im(self, image, filename, conversion=cv2.COLOR_RGB2BGR):
        """
        The function converts image to GBR 
        and saves the image to filename (filename should include the path)

        filename: path to save in including the filename and the extension (jpeg, png, ...)
        conversion: takes any cv2 color flag. Available flags can be checked 
        by running 
        flags = [i for i in dir(cv2) if i.startswith('COLOR_')] 
        Default cv2.COLOR_RGB2BGR (under the assumption the input image is RGB)
        """
        # scale image to 0-1
        image = (image - image.min()) / (image.max() - image.min())
        # convert image type to uint8 after scaling to 0-255
        image = (image * 255).astype(np.uint8)
        saved = cv2.imwrite(filename, cv2.cvtColor(image, conversion))
        return saved


#     @tf.function(input_signature=[tf.TensorSpec(shape=None)])
#     def random_rotate(self, image):
#         if self.batch:
#             angles = np.random.uniform(
#                 low=-self.rotation_angle_range * 90.0,
#                 high=self.rotation_angle_range * 90.0,
#                 size=self.batch,
#             )
#         else:
#             angles = np.random.uniform(
#                 low=-self.rotation_angle_range * 90.0,
#                 high=self.rotation_angle_range * 90.0,
#                 size=1,
#             )
#         image = tfa.image.rotate(image, angles=angles)
#         return image

#     @tf.function(input_signature=[tf.TensorSpec(shape=None)])
#     def random_hsv_transform(self, image):
#         """
#         https://www.tensorflow.org/addons/api_docs/python/tfa/image/random_hsv_in_yiq
#         """
#         max_delta_hue = 2
#         lower_saturation = 0.5
#         upper_saturation = 1
#         lower_value = 0.5
#         upper_value = 1
#
#         image = tfa.image.random_hsv_in_yiq(
#             image,
#             max_delta_hue=max_delta_hue,
#             lower_saturation=lower_saturation,
#             upper_saturation=upper_saturation,
#             lower_value=lower_value,
#             upper_value=upper_value,
#         )
#
#         return image

#     @tf.function(input_signature=[tf.TensorSpec(shape=None)])
#     def random_affine(self, image):
#         shear_x = np.random.uniform(
#             low=-self.affine_trnsfm_range, high=self.affine_trnsfm_range
#         )
#         shear_y = np.random.uniform(
#             low=-self.affine_trnsfm_range, high=self.affine_trnsfm_range
#         )
#         transforms = tf.convert_to_tensor(
#             [1.0, shear_x, 0.0, 0.0, 1.0, shear_y, 0.0, 0.0], dtype=tf.float32
#         )
#         image = tfa.image.transform(image, transforms=transforms)
#         return image
