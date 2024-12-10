from model import ASVFI
from dataset import UCF
from dataset import get_loader

import os
import time
import config
import shutil
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
# import wandb

import torch
from torch.optim import Adam, AdamW
# print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())


# args, unparsed = config.get_args()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--loss', type=str, default='1*L1')
parser.add_argument('--lr', type=float, default=6e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--random_seed', type=int, default=12345)
args = parser.parse_args()

torch.manual_seed(args.random_seed)

train_loader = get_loader('train', batch_size=1, shuffle=True, num_workers=0)
test_loader = get_loader('test', batch_size=1, shuffle=False, num_workers=0)

model = ASVFI().cuda()
criterion = torch.nn.L1Loss()

optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
# if args.optimizer == 'adamw':
#     optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
"""
Problem: Our model is too complex to fit such small dataset(subset of UCF), it overfits early
Solution: 
    What to Experiment Next:
        - 
    Figured:
    - 12/4: Check if more epochs do not lead to better result for Adam -> overfits, lower Flolips, VFIPS, FVD
    - 12/5: Cyclic lr max lr to be ~ ne-4
            Manually lower learning rate during the epoch is good
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=7e-4, step_size_up=1000)
WEIGHT_PATH = f'weights/asvfi/{args.optimizer}_{args.lr}.pth'
BEST_WEIGHT_PATH = 'weights/asvfi/best_asvfi.pth'


def train(args, epoch, batch_ind, losses, duration):
    """
    Run a single epoch to train the model, logging every 200 steps
    len(train_loader) = ~5400   len(test_loader) = ~1300
    batch_ind: guidance value to have i resume training at batch_ind-1
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
                epoch, i, len(train_loader), loss.item(), time.time()-t, WEIGHT_PATH,flush=True))
            # print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tTime: {:.2f}sec\tSaving to {}'.format(
            #     epoch, i+1, 11, loss.item(), time.time()-t, WEIGHT_PATH,flush=True))
            plt.plot(batches, losses['train_losses'], label='Train Loss')
            plt.xlabel('Batch Index')
            plt.ylabel('Loss')
            plt.title(f"Average train_loss: {(sum(losses['train_losses']) / len(losses['train_losses'])):.4f}")
            plt.savefig(f'figs/{args.optimizer}_{args.lr}_train_loss.png')
            plt.close()
            
            #################### Test Logging ####################
            test_loss = test(epoch, test_loader)
            losses['test_losses'].append(test_loss.cpu())
            scheduler.step(test_loss)
            plt.plot(batch_inds, losses['test_losses'], label='Test Loss')
            plt.xlabel('Batch Index')
            plt.ylabel('Loss')
            plt.title(f"Average test_loss: {(sum(losses['test_losses']) / len(losses['test_losses'])):.4f}")
            plt.savefig(f'figs/{args.optimizer}_{args.lr}_test_loss.png')
            plt.close()

            #################### Saving checkpoint ####################
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'batch_ind': batch_ind,
                'duration': duration + (time.time()-t),      # Duration can only be updated outside the train function
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
        # losses.append(loss)
    
        # if (i+1) % 100 == 0 or (i+1) == len(loader):
        #     curr_duration = time.time()-t
        #     t = time.time()
        #     print('Test Epoch: {} [{}/{}]\tLoss: {:.6f}\tTime: {:.2f}sec'.format(
        #         epoch+1, i, len(loader), loss.item(), curr_duration, flush=True))
        #     cum_time += curr_duration
    cum_time = (time.time()-t)
    avg_test_loss = test_loss/len(loader)
    print(f'Test Epoch: {epoch}\tLoss: {avg_test_loss.item():.4f}\tTime: {cum_time:4f}')

    return avg_test_loss


# from torch.utils.data import DataLoader, Subset, Dataset
# from sklearn.model_selection import KFold
# def cv():
#     k_folds = 5
#     kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
#     ucf = UCF()

#     for fold, (train_idx, test_idx) in enumerate(kf.split(ucf)):
#         print(f"Fold {fold + 1}/{k_folds}")

#         train_subset = Subset(ucf, train_idx)
#         test_subset = Subset(ucf, test_idx)

#         train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
#         test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
#         for epoch in range(args.start_epoch, args.max_epoch):
#             train(args, epoch, train_loader)

#         for epoch in range(args.start_epoch, args.max_epoch):
#             val_loss = test(args, epoch, test_loader)

#             print(f'Fold: {fold}, val_loss: {val_loss:.4f}')



def main(args):
    
    batch_ind = 0
    duration = 0
    epoch = args.start_epoch
    losses = {'train_losses': [], 'test_losses': []}
    
    # for epoch in range(args.start_epoch, args.max_epoch):
    # min_test_loss = float('inf')
    if os.path.exists(WEIGHT_PATH):
        print('Loading the saved weights...')
        # checkpoint = torch.load(WEIGHT_PATH, weights_only=True)
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
        # if batch_ind == 0:  
        #     duration = 0    # Reset counting time for new epoch
        print(f"epoch: {epoch}, batch_ind: {batch_ind}, train_losses: {len(losses['train_losses'])} test_losses: {len(losses['test_losses'])}")

    train(args, epoch, batch_ind, losses, duration)   # Epoch, batch_ind updated
    # while True:     # Run a single epoch per run
    #     # batches = range(i+1)
    #     # plt.plot(batches, losses, label='Test Loss')
    #     # plt.xlabel('Batch Index')
    #     # plt.ylabel('Loss')
    #     # plt.savefig('figs/test_loss_per_batch.png')
    #     # plt.close()
    #     break
    
    # Access the parameter groups
    # for i, param_group in enumerate(optimizer.param_groups):
    #     print(f"Parameter Group {i}:")
        
    #     # Print the learning rate
    #     print(f"  Learning Rate: {param_group['lr']}")
        
    #     # Print weight decay, momentum, or other optimizer-specific parameters
    #     if 'weight_decay' in param_group:
    #         print(f"  Weight Decay: {param_group['weight_decay']}")
    #     if 'momentum' in param_group:
    #         print(f"  Momentum: {param_group['momentum']}")

    
    # print(f'-------------------Test Loss: {test_loss:.6f}-------------------')
    
    # if test_loss < min_test_loss:
    #     min_test_loss = test_loss.item()
    #     print(f'Current best: {min_test_loss:.6f} at Epoch: {epoch+1}\t Saved to {BEST_WEIGHT_PATH}')
    #     shutil.copy(WEIGHT_PATH, BEST_WEIGHT_PATH)
    # scheduler.step(test_loss)

if __name__ == "__main__":
    
    main(args)
    # plt.plot(range(2), a)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.savefig('Train_loss.png')

    # a = [[torch.randn(1, 3)]]
    # print(torch.cat(torch.tensor(a)).shape)
    # print(torch.cat(a).shape)