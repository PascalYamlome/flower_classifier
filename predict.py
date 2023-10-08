import numpy as np
import torch
from torchvision import datasets, transforms
import os 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import argparse
from flower_classifier_utils import  load_model, predict



def get_predict_input_args():
    """
    Retrieves and parses command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined the command line arguments. If 
    the user fails to provide some or all , then the default 
    values are used for the missing arguments. 
    Command Line Arguments:

    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    #print("_____________________________________")
    #print("this is the get_input_args() function")
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    
    parser.add_argument('image_path', type = str, default = '/lustre/home/yamlomep/data/flowers/test/3/image_06634.jpg', help = 'sample')
    parser.add_argument('--checkpoint', type = str, default = '/lustre/home/yamlomep/data/flowers/models/trained_vgg_info.pth', help = 'path to model checkpoint')
    parser.add_argument('--top_k', type= int, default=1, help = 'numper classes with highest probability')
    parser.add_argument('--gpu',  action="store_true", help = 'Train on GPU? default is False unless specified')
    parser.add_argument('--category_names', type= str, default='cat_to_name.json', help= 'maps index to actual flower names')


    return parser.parse_args()





def main():

    args = get_predict_input_args()


    top_k = args.top_k
    model_chpt = args.checkpoint
    input_img_path = args.image_path
    use_gpu = args.gpu

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    model,  _, _, class_to_idx =load_model(model_chpt)

    index_to_class = {value: key for key, value in class_to_idx.items()}
    probs, classes = predict(input_img_path, model, device, topk=top_k)

    import json

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)


    print('prob \t', 'class_name')
    for prob, class_index in zip(probs, classes):


        classname = cat_to_name[index_to_class[class_index]]
        print(f'{round(prob,3)}, \t {classname}')

if __name__ == "__main__":
    main()





