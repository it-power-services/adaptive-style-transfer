# A Style-Aware Content Loss for Real-time HD Style Transfer

This repo is a forked from [CompVis/adaptive-style-transfer](https://github.com/CompVis/adaptive-style-transfer), with the aim to upgrade the code to Tensorflow 2

### Original paper and code
***Artsiom Sanakoyeu\*, Dmytro Kotovenko\*, Sabine Lang, Björn Ommer*, In ECCV 2018 (Oral)**

**Website**: https://compvis.github.io/adaptive-style-transfer   
**Paper**: https://arxiv.org/abs/1807.10201

## Pipeline
![pipeline](https://compvis.github.io/adaptive-style-transfer/images/eccv_pipeline_diagram_new_symbols_v2_4.jpg "Method pipeline")

## Example
[![example](https://compvis.github.io/adaptive-style-transfer/images/adaptive-style-transfer_chart_1800px.jpg "Stylization")](https://compvis.github.io/adaptive-style-transfer/images/adaptive-style-transfer_chart.jpg)
Please click on the image for a [high-res version](https://compvis.github.io/adaptive-style-transfer/images/adaptive-style-transfer_chart.jpg).

## Requirements
* Python 3.6
* Tensorflow 2.1
* For the detailed Python packages see [tf2-requirements.txt](https://github.com/it-power-services/adaptive-style-transfer/blob/migrate_to_tf2/tf2-requirements.txt)

## Usage
* The code can be used by running the python script with the needed arguments
`python style_transfer_main.py [arguments]`
* Whether the code is used for training or stylization is specified by the argument `--phase`.
* For all available arguments run `python style_transfer_main.py --help`
	* some of the arguments are phase specific, which can be seen in the square brackets in th description of the argument

### Training
* For training the argument `--phase` should be set to `learn`  
i.e., `python style_transfer_main.py --phase learn`
* important arguments for the training process are:
	* `--style_name` the identifier of the style to be trained, this will be used as a suffix to the name of the containing folder (e.g. 'picasso')
	* `--ptad` path to art dataset
	* `--ptcd` path to content dataset
* other training arguments include a more model specific parameters like the number of filters, the weight of the different losses, etc. Please refer to the help for more details.

### Inference 
Assuming you have already trained a model (or downloaded a pre-trained model)
* For stylizing a new image the argument `--phase` should be set to `stylize`
* The argument `ii_dir` can be used to point to a specific image to stylize (e.g., `path_to_image/image.jpg`), or the path that includes multiple images to stylize. By default this argument is set to `./stylize/input/`.
* The argument `save_dir` should point to the directory in which the stylized images are saved. By default this argument is set to `./stylize/output/`.

#### Pretrained models
[[**TODO**]] 

## Reference

If you use this code or data, please cite the paper:
```
@conference{sanakoyeu2018styleaware,
  title={A Style-Aware Content Loss for Real-time HD Style Transfer},
  author={Sanakoyeu, Artsiom, and Kotovenko, Dmytro, and Lang, Sabine, and Ommer, Bj\"orn},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

### Copyright
```
Adaptive Style Transfer with Tensorflow 2
Copyright (C) 2020  IT Power Services Data Science Team

Adaptive Style Transfer (paper and Tensorflow 1 Code) 
Copyright (C) 2018  Artsiom Sanakoyeu, Dmytro Kotovenko  

Adaptive Style Transfer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
