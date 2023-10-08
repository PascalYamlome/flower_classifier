import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import os
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__

def load_model(model_path_info):

    model_info = torch.load( model_path_info)
    epoch =model_info['epoch']
    model = model_info['model_class']
    optimizer =  model_info['optimizer_state_dict']
    model.load_state_dict(model_info['model_state_dict'])
    class_to_idx = model_info['class_to_idx']
    print('model loaded successfuly')


    return model, optimizer, epoch, class_to_idx




def buid_CNN_model(num_classes=102, model_architecture = 'vgg', TL = True):

    if model_architecture.lower() =='vgg':
         #You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights

        if TL:
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            model = models.vgg16()


        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)


    elif model_architecture.lower() =='resnet':
        

        if TL:
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = models.resnet18()
        
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    elif model_architecture.lower() =='alexnet':
        

        if TL:
            model = models.alexnet(weights='AlexNet_Weights.DEFAULT') #You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights
            for param in model.features.parameters():
                param.requires_grad = False
            
        else:
            model = models.alexnet() 

        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)


    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
   # define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(image)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    return img_tensor

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    '''
    
    pill_img = Image.open(image_path)
    img_tensor = process_image(pill_img)

    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    #model = models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.to(device)
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor.to(device))

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)
    
    pobs, classes = output.softmax(dim= 1).topk(topk)
    
    return pobs.detach().cpu().squeeze().numpy().tolist(), classes.detach().cpu().squeeze().numpy().tolist()
    # TODO: Implement the code to predict the class from an image file