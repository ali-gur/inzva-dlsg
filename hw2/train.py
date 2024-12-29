import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import train_arg_parser, set_seed, save_model
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score


def train_model(model, train_loader, val_loader, optimizer, criterion, args, save_path):
    '''
    Trains the given model over multiple epochs, tracks training and validation losses, 
    and saves model checkpoints periodically.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration (e.g., epochs, batch size, device).
    - save_path (str): Directory path to save model checkpoints and training history.

    Functionality:
    - Creates directories to save results and checkpoints.
    - Calls `train_one_epoch` to train and validate the model for each epoch.
    - Saves model checkpoints every 5 epochs.
    - Plots the training and validation loss curves and the Dice coefficient curve.
    '''
   

    train_loss_history = []
    val_loss_history = []
    dice_coef_history = []

    for epoch in range(args.epoch):
        train_one_epoch(model, 
                        train_loader, 
                        val_loader, 
                        train_loss_history, 
                        val_loss_history, 
                        dice_coef_history, 
                        optimizer, 
                        criterion, 
                        args, 
                        epoch, 
                        save_path)
    
    plot_train_val_history(train_loss_history, val_loss_history, save_path, args)
    plot_metric(dice_coef_history, 'Dice Coefficient', save_path, args, 'dice_coef')

    

def train_one_epoch(model, train_loader, val_loader, train_loss_history, val_loss_history, 
                    dice_coef_history, optimizer, criterion, args, epoch, save_path):
    '''
    Performs one full epoch of training and validation, computes metrics, and visualizes predictions.

    Args:
    - model (torch.nn.Module): The neural network model to train.
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - train_loss_history (list): List to store the average training loss per epoch.
    - val_loss_history (list): List to store the average validation loss per epoch.
    - dice_coef_history (list): List to store the Dice coefficient per epoch.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
    - criterion (torch.nn.Module): The loss function used for training.
    - args (argparse.Namespace): Parsed arguments containing training configuration.
    - epoch (int): The current epoch number.
    - save_path (str): Directory path to save visualizations and model checkpoints.

    Functionality:
    - Sets the model to training mode and performs a forward and backward pass for each batch in the training data.
    - Computes the training loss and updates the weights.
    - Sets the model to evaluation mode and computes validation loss and Dice coefficients.
    - Visualizes predictions periodically and saves them to the specified directory.
    - Appends the average training and validation losses, and the Dice coefficient to their respective lists.
    - Prints the Dice coefficient and loss values for the current epoch.
    '''
    print(f"Epoch {epoch+1}/{args.epoch}")
    model.train()
    train_loss = 0.0
    print(len(train_loader))
    for batch in train_loader:
        inputs, targets = batch

        # forward pass
        optimizer.zero_grad()
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs, targets)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # compute metrics
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)

    model.eval()
    val_loss = 0.0
    dice_coef = 0.0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch

            # forward pass
            outputs = model(inputs)

            # compute loss
            loss = criterion(outputs, targets)

            # compute metrics
            val_loss += loss.item()

            # compute dice coefficient
            dice_coef += compute_dice_score(outputs, targets)
           
    
    if (epoch + 1) % 1 == 0:
        save_model(model, save_path)
        images, masks, outputs = inputs[:3], targets[:3], outputs[:3]
        visualize_predictions(images, masks, outputs, save_path, epoch, 0)
            
    
    val_loss /= len(val_loader)
    dice_coef /= len(val_loader)
    val_loss_history.append(val_loss)
    dice_coef_history.append(dice_coef)



    





if __name__ == '__main__':
    args = train_arg_parser()
    save_path = "results"
    set_seed(42)

    # Define dataset
    dataset = MadisonStomach(data_path="madison-stomach", mode=args.mode)

    # Define train and val indices

    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    # Define Subsets of to create trian and validation dataset
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Define dataloader
    train_loader = DataLoader(train_subset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.bs, shuffle=False)

    # Define your model
    model = UNet(in_channels=3, out_channels=1)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    train_model(model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                save_path=save_path)


