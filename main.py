from model import ASVFI
from dataset import UCF
from dataset import get_loader

import os
import time
import config
import shutil
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam, AdamW

import argparse
# args, unparsed = config.get_args() # Keep this for now, will decide later if it's essential

parser = argparse.ArgumentParser(description='ASVFI Model Training')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Epoch to start training from')
parser.add_argument('--loss', type=str, default='1*L1',
                    help='Loss function configuration')
parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='Adam optimizer beta1')
parser.add_argument('--beta2', type=float, default=0.99,
                    help='Adam optimizer beta2')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='Optimizer type (e.g., adamw)')
parser.add_argument('--random_seed', type=int, default=12345,
                    help='Random seed for reproducibility')
args = parser.parse_args()

torch.manual_seed(args.random_seed)

train_loader = get_loader('train', batch_size=1, shuffle=True, num_workers=0)
test_loader = get_loader('test', batch_size=1, shuffle=False, num_workers=0)

model = ASVFI().cuda()
criterion = torch.nn.L1Loss()

optimizer = Adam(model.parameters(), lr=args.lr,
                 betas=(args.beta1, args.beta2))
# """
# Problem: Our model is too complex to fit such small dataset(subset of UCF), it overfits early
# Solution:
#     What to Experiment Next:
#         -
#     Figured:
#     - 12/4: Check if more epochs do not lead to better result for Adam -> overfits, lower Flolips, VFIPS, FVD
#     - 12/5: Cyclic lr max lr to be ~ ne-4
#             Manually lower learning rate during the epoch is good
# """
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=1)
WEIGHT_PATH = f'weights/asvfi/{args.optimizer}_{args.lr}.pth'
BEST_WEIGHT_PATH = 'weights/asvfi/best_asvfi.pth'


def plot_loss(args, batches, losses, loss_type):
    """
    Plots the loss curve and saves it to a file.

    Args:
        args: Command line arguments.
        batches: List of batch indices.
        losses: List of loss values.
        loss_type: Type of loss (e.g., 'train', 'test').
    """
    plt.plot(batches, losses, label=f'{loss_type.capitalize()} Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title(f"Average {loss_type}_loss: {(sum(losses) / len(losses)):.4f}")
    plt.savefig(f'figs/{args.optimizer}_{args.lr}_{loss_type}_loss.png')
    plt.close()


def train(args, epoch, batch_ind, losses, duration):
    """
    Runs a single epoch to train the model.

    Args:
        args: Command line arguments.
        epoch (int): Current epoch number.
        batch_ind (int): Batch index to resume training from.
        losses (dict): Dictionary to store training and testing losses.
        duration (float): Total training duration so far.
    """
    print(f'Start training from batch: {batch_ind}')
    model.train()
    t = time.time()
    batch_inds = []
    for i, batch in enumerate(train_loader):

        if i < batch_ind:  # Skip to the previously saved batch index
            continue

        vid, mfcc, gt = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

        optimizer.zero_grad()
        pred = model(vid, mfcc)

        pred = torch.cat(pred)
        gt = torch.cat([gt])

        loss = criterion(pred, gt)
        losses['train_losses'].append(loss.item())
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0 or (i+1) == len(train_loader):
            batch_ind = i+1                   # For train logging
            batches = range(batch_ind)

            batch_inds.append(i+1)            # For test logging
            if (i+1) == len(train_loader):    # Reset to initial condition after finishing an epoch
                epoch += 1
                batch_ind = 0

            #################### Train Logging ####################
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tTime: {:.2f}sec\tSaved to {}'.format(
                epoch, i, len(train_loader), loss.item(), time.time()-t, WEIGHT_PATH, flush=True))
            plot_loss(args, batches, losses['train_losses'], 'train')

            #################### Test Logging ####################
            test_loss = test(epoch, test_loader)
            losses['test_losses'].append(test_loss.cpu())
            scheduler.step(test_loss)
            plot_loss(args, batch_inds, losses['test_losses'], 'test')

            #################### Saving checkpoint ####################
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'batch_ind': batch_ind,
                # Duration can only be updated outside the train function
                'duration': duration + (time.time()-t),
            }
            torch.save(checkpoint, WEIGHT_PATH)


def test(epoch, loader):
    model.eval()
    t = time.time()
    cum_time = 0
    test_loss = 0
    for i, batch in enumerate(loader):

        vid, mfcc, gt = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

        with torch.no_grad():
            pred = model(vid, mfcc)

        pred = torch.cat(pred)
        gt = torch.cat([gt])

        loss = criterion(pred, gt)
        test_loss += loss

    cum_time = (time.time()-t)
    avg_test_loss = test_loss/len(loader)
    print(
        f'Test Epoch: {epoch}\tLoss: {avg_test_loss.item():.4f}\tTime: {cum_time:4f}')

    return avg_test_loss


def test(epoch, loader):
    """
    Evaluates the model on the test dataset.

    Args:
        epoch (int): Current epoch number.
        loader (torch.utils.data.DataLoader): DataLoader for the test dataset.

    Returns:
        torch.Tensor: Average test loss.
    """
    model.eval()
    t = time.time()
    cum_time = 0
    test_loss = 0
    for i, batch in enumerate(loader):

        vid, mfcc, gt = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()

        with torch.no_grad():
            pred = model(vid, mfcc)

        pred = torch.cat(pred)
        gt = torch.cat([gt])

        loss = criterion(pred, gt)
        test_loss += loss

    cum_time = (time.time()-t)
    avg_test_loss = test_loss/len(loader)
    print(
        f'Test Epoch: {epoch}\tLoss: {avg_test_loss.item():.4f}\tTime: {cum_time:4f}')

    return avg_test_loss


def main(args):
    """
    Main function to orchestrate the training process.

    Args:
        args: Command line arguments.
    """
    batch_ind = 0
    duration = 0
    epoch = args.start_epoch
    losses = {'train_losses': [], 'test_losses': []}

    if os.path.exists(WEIGHT_PATH):
        print('Loading the saved weights...')
        checkpoint = torch.load(WEIGHT_PATH)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        batch_ind = checkpoint['batch_ind']
        duration = checkpoint['duration']
        """
        Need to modify saving losses for epoch > 1, consider AverageMeter
        """
        print(
            f"epoch: {epoch}, batch_ind: {batch_ind}, train_losses: {len(losses['train_losses'])} test_losses: {len(losses['test_losses'])}")

    train(args, epoch, batch_ind, losses, duration)


if __name__ == "__main__":

    main(args)
