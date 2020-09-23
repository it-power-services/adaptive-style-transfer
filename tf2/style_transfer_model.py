import tensorflow as tf
import os
import time
import pathlib

from style_transfer_lib import (
    #    options,
    read_image,
    get_batch,
    batch_to_one_image,
    Discriminator,
    Generator,
    discriminator_loss,
    generator_loss,
)

from style_transfer_img_aug import Augmentor


class model_ast:
    def __init__(self, options):
        self.options = options
        # ---------------------
        tf.print("Initializing the model", options.style_name)
        self.path_to_content_images = options.path_to_content_dataset
        self.path_to_paintings = options.path_to_art_dataset
        images_path = pathlib.Path(self.path_to_content_images)
        self.images_dataset = [str(x) for x in images_path.glob("**/*.jpg")]
        paintings_path = pathlib.Path(self.path_to_paintings)
        self.paintings_dataset = [str(x) for x in paintings_path.glob("**/*.jpg")]

        tf.print("Found ", len(self.images_dataset), " images")
        tf.print("Found ", len(self.paintings_dataset), " paintings")
        # ----------------------
        self.path_to_model = os.path.join(
            options.path_to_models, "model_" + options.style_name
        )
        self.path_to_samples = os.path.join(self.path_to_model, "sample_batches")
        #        self.path_to_generated = os.path.join(self.path_to_model, "stylized_images")
        self.path_to_logs = os.path.join(self.path_to_model, "logs_ast")
        self.path_to_checkpoint = os.path.join(
            self.path_to_model, "training_checkpoints"
        )
        self.path_to_checkpoint_long = os.path.join(
            self.path_to_model, "training_checkpoints_long"
        )
        # ----------------------
        # Create all the folders for saving the model
        if not os.path.exists(options.path_to_models):
            os.makedirs(options.path_to_models)
        if not os.path.exists(options.in_images_dir):
            os.makedirs(options.in_images_dir)
        if not os.path.exists(options.out_images_dir):
            os.makedirs(options.out_images_dir)
        # --
        if not os.path.exists(self.path_to_samples):
            os.makedirs(self.path_to_samples)
        #         if not os.path.exists(self.path_to_generated):
        #             os.makedirs(self.path_to_generated)
        if not os.path.exists(self.path_to_model):
            os.makedirs(self.path_to_model)
        if not os.path.exists(self.path_to_logs):
            os.makedirs(self.path_to_logs)
        if not os.path.exists(self.path_to_checkpoint):
            os.makedirs(self.path_to_checkpoint)
        if not os.path.exists(self.path_to_checkpoint_long):
            os.makedirs(self.path_to_checkpoint_long)
        # ----------------------
        self.summary_writer = tf.summary.create_file_writer(
            os.path.join(self.path_to_logs, "fit/" + time.strftime("%Y%m%d-%H%M%S"))
        )
        # ----------------------
        self.generator = Generator(input_shape=[None, None, 3])
        self.discriminator = Discriminator(input_shape=[None, None, 3])
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=options.lr, beta_1=0.5
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=options.lr, beta_1=0.5
        )
        # ----------------------
        self.checkpoint_name = "ckpt"
        #
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        self.checkpoint_mngr = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self.path_to_checkpoint,
            checkpoint_name=self.checkpoint_name,
            max_to_keep=5,
        )
        # checkpoint long (save checkpoint every 50,000 step for result comparison)
        self.checkpoint_long = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )
        self.checkpoint_long_mngr = tf.train.CheckpointManager(
            self.checkpoint_long,
            directory=self.path_to_checkpoint_long,
            checkpoint_name=self.checkpoint_name,
            max_to_keep=None,
        )
        # ----------------------
        self.augmentor = Augmentor(crop_size=(options.image_size, options.image_size))
        # ----------------------
        # load chekpoint
        self.start_step = self.load_ckpt(self.options.ckpt_nmbr)

    # --
    def load_ckpt(self, ckpt_nr="auto"):
        """
        Loads checkpoint and return step/iteration_number to continue from or to use for stylization

        Loads checkpoint ckpt_nr if it exists in path_to_checkpoint or path_to_checkpoint_long
        else tries to load the latest saved checkpoint in path_to_checkpoint, 
        if none exists returns 0 as a starting step

        ckpt_nr: integer
        """
        # checkpoint_mngr dosn't have restore_or_initialize in tf 2.1
        # (the available version in the DS server)
        # ckpt = self.checkpoint_mngr.restore_or_initialize()
        loaded = False
        if ckpt_nr != "auto":
            try:
                # try loading from checkpoint
                ckpt = os.path.join(
                    self.path_to_checkpoint, self.checkpoint_name + "-" + str(ckpt_nr)
                )
                tf.print("Trying to load the checkpoint from", ckpt)
                self.checkpoint.restore(ckpt).assert_existing_objects_matched()
                loaded = True
                tf.print("Checkpoint loaded")
                start_step = int(ckpt_nr) + 1
            except:
                try:
                    # try loading from checkpoint_long
                    ckpt = os.path.join(
                        self.path_to_checkpoint_long,
                        self.checkpoint_name + "-" + str(ckpt_nr),
                    )
                    tf.print("Trying to load the checkpoint from", ckpt)
                    self.checkpoint_long.restore(ckpt).assert_existing_objects_matched()
                    loaded = True
                    tf.print("Checkpoint loaded")
                    start_step = int(ckpt_nr) + 1
                except:
                    tf.print("Loading checkpoint", str(ckpt_nr), "failed")
                    ckpt = self.checkpoint_mngr.latest_checkpoint
                    pass
        else:
            ckpt = self.checkpoint_mngr.latest_checkpoint
        if not loaded:
            if ckpt is not None:
                tf.print("Loading the latest checkpoint from", ckpt)
                start_step = int(ckpt.split("-")[-1]) + 1
                self.checkpoint.restore(ckpt).assert_existing_objects_matched()
                tf.print("Checkpoint loaded")
            else:
                tf.print("No checkpoint is loaded, starting from the first iteration")
                start_step = 0
        return start_step

    # --
    @tf.function
    def train_step(self, input_image, input_painting, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            tf.print("Passing through the generator")
            gen_outputs = self.generator([input_image], training=True)
            input_features = gen_outputs["input_features"]
            output_image = gen_outputs["output_image"]
            output_features = gen_outputs["output_features"]
            # --
            tf.print("Passing through the discriminator")
            input_image_discr_predictions = self.discriminator(
                [input_image], training=True
            )
            input_painting_discr_predictions = self.discriminator(
                [input_painting], training=True
            )
            output_image_discr_predictions = self.discriminator(
                [output_image], training=True
            )
            # --
            tf.print("Computing generator loss")
            gen_total_loss = generator_loss(
                output_discr_preds=output_image_discr_predictions,
                input_image=input_image,
                output_image=output_image,
                output_image_features=output_features,
                input_image_features=input_features,
                options=self.options,
            )
            tf.print("Computing discriminator loss")
            disc_total_loss = discriminator_loss(
                input_painting_discr_predictions=input_painting_discr_predictions,
                input_image_discr_predictions=input_image_discr_predictions,
                output_image_discr_predictions=output_image_discr_predictions,
            )
        # --
        generator_gradients = gen_tape.gradient(
            gen_total_loss, self.generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_total_loss, self.discriminator.trainable_variables
        )
        # --
        tf.print("Applying gradients")
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )
        tf.print("Discriminator loss: ", disc_total_loss)
        tf.print("Generator loss: ", gen_total_loss)
        tf.print(" ---------------------------------------- ")
        # --
        with self.summary_writer.as_default():
            tf.summary.scalar("gen_total_loss", gen_total_loss, step=step)
            tf.summary.scalar("disc_total_loss", disc_total_loss, step=step)

    # --
    def fit(self):
        # start training
        tf.print("Start Training .... ")
        for step in range(self.start_step, self.options.total_steps):
            start = time.time()
            tf.print("step: ", step)
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            # --
            images_batch = get_batch(
                self.images_dataset,
                augmentor=self.augmentor,
                batch_size=self.options.batch_size,
            )
            # images_batch = self.augmentor(images_batch)

            paintings_batch = get_batch(
                self.paintings_dataset,
                augmentor=self.augmentor,
                batch_size=self.options.batch_size,
            )
            # paintings_batch = self.augmentor(paintings_batch)

            self.train_step(images_batch, paintings_batch, step)
            # --
            # saving batch images every 500 steps
            if (step % 500) == 0:
                # save batch images
                images_batch_one = batch_to_one_image(images_batch)
                paintings_batch_one = batch_to_one_image(paintings_batch)
                img = tf.concat([images_batch_one, paintings_batch_one], 1)
                self.augmentor.save_im(
                    image=img.numpy(),
                    filename=os.path.join(
                        self.path_to_samples, "batch_" + str(step.numpy()) + ".jpg"
                    ),
                )
                # --
                generated = self.generator([images_batch])["output_image"]
                generated_with_input = tf.concat(
                    [images_batch_one, batch_to_one_image(generated)], 1
                )
                self.augmentor.save_im(
                    image=generated_with_input.numpy(),
                    filename=os.path.join(
                        self.path_to_samples, str(step.numpy()) + "_generated.jpeg"
                    ),
                )

            # saving (checkpoint) the model every 2000 steps
            if (step != 0) and ((step % 2000) == 0):
                saved_ckpt = self.checkpoint_mngr.save(checkpoint_number=step)
                tf.print(saved_ckpt, ": Saved")
            # saving in checkpoint_long every 50,000 steps (for result comparison)
            if (step != 0) and ((step % 50000) == 0):
                saved_ckpt = self.checkpoint_long_mngr.save(checkpoint_number=step)
                tf.print(saved_ckpt, ": Saved")
            # --
            tf.print(
                "Time taken for step {} is {} sec\n".format(step, time.time() - start)
            )
        self.checkpoint_mngr.save(checkpoint_number=step)

    # --
    def stylize(self, image):
        """
        image: could be path to image or an image of shape (h, w, c) as an array or tensor
        """

        if isinstance(image, str):
            basename = os.path.split(image)[-1]
            filename = os.path.splitext(basename)[0]
            image = read_image(image)
        else:
            filename = time.strftime("%Y%m%d-%H%M%S")

        # Resize the smallest side of the image to self.options.image_size
        # (optional, the need for scaling depends on the available memory)
        h, w, c = image.shape

        if min(h, w) > self.options.image_size:
            scale_ratio = self.options.image_size / min(h, w)
            image_scaled = tf.image.resize(
                image,
                size=[int(h * scale_ratio), int(w * scale_ratio)],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
        else:
            image_scaled = image

        ckpt_nr = self.start_step

        if ckpt_nr == 0:
            raise Exception("No checkpoint is found")
        tf.print(
            "Style",
            self.options.style_name + ",",
            "checkpoint number",
            str(ckpt_nr - 1),
        )

        image_generated = self.generator([image_scaled[tf.newaxis, ...]])[
            "output_image"
        ]
        image_generated = image_generated.numpy()[0, :, :, :]

        # scale image back to original size
        image_generated = tf.image.resize(
            image_generated, size=[h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        image_generated = image_generated.numpy()

        # save generated image
        save_to = os.path.join(
            self.options.out_images_dir,
            filename
            + "_ckpt"
            + str(ckpt_nr - 1)
            + "_"
            + self.options.style_name
            + ".jpeg",
        )
        saved = self.augmentor.save_im(image=image_generated, filename=save_to)
        if saved:
            tf.print("Stylized image saved to", save_to)
        else:
            tf.print("[!] Generated image was not saved")
