# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

scripts:
predict.py
train.py
flower_classifier_utils.py

# predict.py
This Python script uses a pre-trained neural network to classify flowers based on the input image.

The script supports the following command line arguments:

image_path (positional argument): Path to the input image (default: '/lustre/home/yamlomep/data/flowers/test/3/image_06634.jpg').

--checkpoint: Path to the model checkpoint file (default: '/lustre/home/yamlomep/data/flowers/models/trained_vgg_info.pth').

--top_k: Number of classes with the highest probability to display (default: 1).

--gpu: Use GPU for inference (default: False). If specified, the script will use GPU if available.

--category_names: Path to the file mapping index to actual flower names (default: 'cat_to_name.json').

besure to change image_path and checkpoint default values before use



Usage
To use the flower classifier script, follow these steps:

Ensure you have the necessary dependencies installed. See the "Dependencies" section for details.

Run the script with the desired input image using the following command:



# train.py
Command Line Arguments
The script supports the following command line arguments for customization:

--data_dir: Base directory containing the training data (default: '/lustre/home/yamlomep/data/flowers').

--save_dir: Directory to save the model checkpoint files (default: '/lustre/home/yamlomep/data/flowers/models').

--arch: Model architecture (default: 'vgg').

--learning_rate: Learning rate for training (default: 0.001).

--epochs: Number of training epochs (default: 20).

--gpu: Train using GPU if available (default: False). Specify this flag to use GPU.

--save_interval: Save model weights every _ epochs (default: 10).

--transfer_learn: Perform transfer learning (default: False). Specify this flag to perform transfer learning.

--resume: Resume training from a previously saved checkpoint (default: False). Specify this flag to resume training.

Authors

# flower_classifier_utils.py
 