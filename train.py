import numpy as np
import torch
from torchvision import datasets, transforms
import os 
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import argparse
from flower_classifier_utils import buid_CNN_model, load_model



def get_train_input_args():
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
    
    parser.add_argument('data_dir', type = str, default = '/lustre/home/yamlomep/data/flowers', help = 'base directory that contains training data')
    parser.add_argument('--save_dir', type = str, default = '/lustre/home/yamlomep/data/flowers/models', help = 'checkpoint save directory')
    parser.add_argument('--arch', type = str, default = 'vgg', help = 'model architecture')
    parser.add_argument('--learning_rate', type = float, default=0.001, help = 'learning rate')
    parser.add_argument('--epochs', type= int, default=20, help = 'Number of training epochs')
    parser.add_argument('--gpu',  action="store_true", help = 'Train on GPU? default is False unless specified')
    parser.add_argument('--save_interval', type= int, default=10, help= 'save model weights every _ epochs')
    parser.add_argument('--transfer_learn',action="store_true", help=' do transfer learning? default value False unless specified')
    parser.add_argument('--resume', action="store_true", help=' resume traiing ')
    # parser.add_argument('--outputfolder', type=str, default = 'mixed_resolution', help='output folder ')
    


    return parser.parse_args()





def create_dataloaders( data_dir, train_batch_size=128, valid_batch_size = 64, shuffle_ = True):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    # TODO: Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225] )])

    test_datatransforms = transforms.Compose([transforms.Resize(255),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225]) ])

    # TODO: Load the datasets with ImageFolder
    #Train data
    train_dataset = datasets.ImageFolder(train_dir, transform=train_data_transforms)

    # validation dataset. here we aply the same transforms as the test image
    validation_dataset = datasets.ImageFolder(valid_dir, transform=test_datatransforms)

    test_dataset = datasets.ImageFolder(test_dir, transform=test_datatransforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle = shuffle_)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=valid_batch_size, shuffle = shuffle_)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=valid_batch_size)

    return train_dataloader, validation_dataloader, test_dataloader, train_dataset.class_to_idx





def train_loop(model, 
               device, 
               train_dataloader, 
               validation_dataloader ,
               learning_rate = 0.001, 
               num_epochs = 21,
               model_chkpt_path = '/lustre/home/yamlomep/data/flowers/models',
               save_interval =20,
               model_arch = 'vgg',
               resume = True,
               class_to_idx = {}):
    
    model_path_info = os.path.join(model_chkpt_path, f'trained_{model_arch}_info.pth')



    # Initialize the model, loss function, and optimizer
    start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)







    if resume:

        print('... Attempting to resume training from previously saved checkpoint')

        #check if checkpoint exist at the specified directory
        if os.path.exists(model_path_info):
            #load 
            model,  optimizer_statedict, start_epoch =load_model(model_path_info)
            optimizer.load_state_dict(optimizer_statedict)
            print(f'{model_arch} model loaded from {model_chkpt_path}.')
            print(f'resuming training from epoch {start_epoch}')
        else:
            print('Warning!: unable to load previously saved model')
            print(f'{model_path_info} does not exist')
            print(f'starting training from eopch {start_epoch}' )
        


    
    

    # Function to calculate accuracy
    def calculate_accuracy(predictions, labels):
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_dataloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (torch.argmax(outputs, 1) == labels).sum().item()

        train_loss = train_loss / len(train_dataloader.dataset)
        train_accuracy = train_correct / len(train_dataloader.dataset)

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0


        with torch.no_grad():
            for images, labels in validation_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()

        val_loss = val_loss / len(validation_dataloader.dataset)
        val_accuracy = val_correct / len(validation_dataloader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        if  epoch >0  and (epoch+1) % save_interval == 0:
            print('saving model')

            

            save_path = os.path.join(model_chkpt_path, f'trained_{model_arch}_info.pth')

            torch.save({
                'model_class':model,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_to_idx': class_to_idx,
                'epoch':epoch,

            }, save_path)

            print(f'saved model info at {save_path}')


    print("Training finished.")

def load_model_and_test(model, device, path, test_dataloader):
    test_correct = 0
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)



            test_correct += (torch.argmax(outputs, 1) == labels).sum().item()

    test_accuracy = 100* test_correct / len(test_dataloader.dataset)
    return test_accuracy





def main():


    args = get_train_input_args()

    base_dir = args.data_dir
    model_arch = args.arch
    use_gpu = args.gpu
    number_of_epochs = args.epochs
    savemodel_chpt_dir = args.save_dir
    save_model_interval = args.save_interval
    learning_rate = args.learning_rate
    resume = args.resume


    #define dataloaders

    train_dataloader, validation_dataloader, test_dataloader, class_to_idx = create_dataloaders( base_dir, 
                                                                                  train_batch_size=128, 
                                                                                  valid_batch_size = 64, 
                                                                                  shuffle_ = True)
    
    



    
    model = buid_CNN_model(model_architecture = model_arch)

    if use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'




    print(f'***************************training on {device}*******************************')


    train_loop(model, 
               device, 
               train_dataloader, 
               validation_dataloader ,
               learning_rate = learning_rate, 
               num_epochs = number_of_epochs,
               model_chkpt_path = savemodel_chpt_dir,
               save_interval =save_model_interval,
               model_arch=model_arch,
               resume = resume,
               class_to_idx= class_to_idx)
    

    model_path = os.path.join(savemodel_chpt_dir, f"trained_{model_arch}.pth")
    test_acc = load_model_and_test(model, device, model_path, test_dataloader)

    print('****************testing model******************')

    print(f'Test accuracy: {test_acc}')
    

if __name__ == "__main__":
    main()












