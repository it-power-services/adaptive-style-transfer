import argparse
import pathlib
from style_transfer_model import model_ast


def parse_list(str_value):
    if "," in str_value:
        str_value = str_value.split(",")
    else:
        str_value = [str_value]
    return str_value


parser = argparse.ArgumentParser(description="")

# ========================== GENERAL PARAMETERS ========================= #
parser.add_argument(
    "--style_name", dest="style_name", default="cezanne", help="Name of the model/style. (Default: cezanne)"
)
parser.add_argument(
    "--phase",
    dest="phase",
    default="learn",
    help="Specify current phase: learn or stylize. (Default: learn)",
)
parser.add_argument(
    "--image_size",
    dest="image_size",
    type=int,
    default=256 * 3,
    help="For training phase: will crop out images of this particular size."
    "For inference phase: each input image will have the smallest side of this size. (Default: 256 * 3)",
)
parser.add_argument(
    "--ptmd",
    dest="path_to_models",
    type=str,
    default="./models",
    help="Path to the models and checkpoints. (Default: ./models)",
)
parser.add_argument(
    "--ckpt_nmbr",
    dest="ckpt_nmbr",
    type=str,
    default="auto",
    help="Checkpoint number we want to use for stylizing, or continue from in training"
    'If "auto" then the latest available will be used. (Default: auto)',
)
# ========================= TRAINING PARAMETERS ========================= #
parser.add_argument(
    "--ptad",
    dest="path_to_art_dataset",
    type=str,
    default="../../data/paul-cezanne/",
    help="[phase: learn] Directory with paintings representing style we want to learn. (Default: ../../data/paul-cezanne/)",
)
parser.add_argument(
    "--ptcd",
    dest="path_to_content_dataset",
    type=str,
    default="../../data/new_models/data_large/",
    help="[phase: learn] Path to the training dataset (images to the learn extracting the content from). (Default: ../../data/new_models/data_large/)",
)
parser.add_argument(
    "--total_steps",
    dest="total_steps",
    type=int,
    default=int(3e5),
    help="[phase: learn] Total number of steps. (Default: 350,000)",
)
parser.add_argument(
    "--batch_size", dest="batch_size", type=int, default=1, help="[phase: learn] # images in batch. (Default: 1)"
)
parser.add_argument(
    "--lr", dest="lr", type=float, default=0.0002, help="[phase: learn] initial learning rate for adam. (Default: 0.0002)"
)
# parser.add_argument('--save_freq',
#                     dest='save_freq',
#                     type=int,
#                     default=1000,
#                     help='Save model every save_freq steps')
parser.add_argument(
    "--ngf",
    dest="gf_dim",
    type=int,
    default=32,
    help="[phase: learn] Number of filters in first conv layer of generator(encoder-decoder). (Default: 32)",
)
parser.add_argument(
    "--ndf",
    dest="df_dim",
    type=int,
    default=64,
    help="[phase: learn] Number of filters in first conv layer of discriminator. (Default: 64)",
)

# Weights of different losses.
parser.add_argument(
    "--dlw",
    dest="discr_loss_weight",
    type=float,
    default=1.0,
    help="[phase: learn] Weight of discriminator loss. (Default: 1.0)",
)
parser.add_argument(
    "--tlw",
    dest="transformer_loss_weight",
    type=float,
    default=100.0,
    help="[phase: learn] Weight of transformer loss. (Default: 100.0)",
)
parser.add_argument(
    "--flw",
    dest="feature_loss_weight",
    type=float,
    default=100.0,
    help="[phase: learn] Weight of feature loss. (Default: 100.0)",
)
# parser.add_argument('--dsr',
#                     dest='discr_success_rate',
#                     type=float,
#                     default=0.8,
#                     help='Rate of trials that discriminator will win on average.')


# ========================= INFERENCE PARAMETERS ========================= #
parser.add_argument(
    "--ii_dir",
    dest="in_images_dir",
    type=str,
    default="./stylized/input/",
    help="[phase: stylize] Directory with images we want to process. (Default: ./stylized/input/)",
)
parser.add_argument(
    "--save_dir",
    dest="out_images_dir",
    type=str,
    default="./stylized/output/",
    help="[phase: stylize] Directory to save inference output images. (Default: ./stylized/output/)",
)

options = parser.parse_args()

mod_ast = model_ast(options)

if options.phase == "learn":
    mod_ast.fit()
if options.phase == "stylize":
    if options.in_images_dir.endswith((".jpg", ".jpeg")):
        images = [options.in_images_dir]
    else:
        images_path = pathlib.Path(options.in_images_dir)
        images = [str(x) for x in images_path.glob("**/*.jpg")]
    for image in images:
        mod_ast.stylize(image)
