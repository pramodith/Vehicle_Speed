import torch
import torch.nn as nn
import torch.optim as optim
from DataHandler import DataHandler, TestDataHandler
from torch.utils.data.dataloader import DataLoader
import sys
import os
import argparse
from torch.nn import functional as F
import numpy as np
from scipy.ndimage import zoom
import cv2
from sklearn.metrics import f1_score, accuracy_score
from torchsummary import summary
class Module(nn.Module):

    def __init__(self, width=256, height=256, save_dir=None):
        super(Module, self).__init__()

        # Directory in which model weights will be saved.
        if save_dir:
            self.save_dir = save_dir

        self.width = width
        self.height = height
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,24,5,2),
            nn.ReLU(),
            nn.Conv2d(24,36,5,2),
            nn.ReLU(),
            nn.Conv2d(36,48,5,2),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.linear_layers = nn.Sequential(nn.Linear(1152,100),
                nn.ReLU(),
                nn.Linear(100,50),
                nn.ReLU(),
                nn.Linear(50,10),
                nn.ReLU(),
                nn.Linear(10,1))


    # Initializes convolutional layers with Xavier initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    # Loads the model with the saved weights file
    def load_model(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

    # Implement the forward pass
    def forward(self, input):
        out = self.conv_layers(input)
        out = out.view(out.shape[0],-1)
        score = self.linear_layers(out)
        return score

    # Trains the DNN, default parameters suggested by paper.
    def train_model(self, train_dir, gt_file_path, batch_size=16, epochs=5, lr=0.0001, momentum=0.9, weight_decay=0.00005):
        # If a GPU is available transfer model to GPU
        if torch.cuda.is_available():
            self.cuda()
        # Use Cross Entropy Loss
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Instantiate data handler and loader to efficiently create batches
        data_handler = DataHandler(train_dir,gt_file_path, mode='train')
        num_workers = 1

        # Use the weighted sampler for training
        loader = DataLoader(data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)
        dev_data_handler = DataHandler(train_dir, gt_file_path, mode='val')
        dev_loader = DataLoader(dev_data_handler, batch_size, True, num_workers=num_workers, pin_memory=True)

        # Store the loss history and create variables that will store the best loss
        batch_loss_histroy = []
        best_dev_loss = sys.maxsize
        best_loss = sys.maxsize

        for epoch in range(epochs):
            print("Epoch is " + str(epoch))

            # Model is in train mode so that gradients are computed
            self.train()
            total_loss = 0
            batch_total_loss = 0

            for i, batch in enumerate(loader):
                images = batch[0]
                labels = batch[1].float()

                # Move all inputs to GPU
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                # Run the forward pass
                score = self.forward(images)

                # Reset the gradients every batch
                optimizer.zero_grad()

                # Compute loss, gradients and backprop
                output = loss(score.squeeze(1), labels)
                output.backward()
                optimizer.step()

                total_loss += output.item()
                batch_total_loss += output.item()

                if i % 50 == 0:
                    batch_loss_histroy.append(output.item())
                    print("Loss: for batch " + str(i) + " is " + str(batch_total_loss))
                    batch_total_loss = 0
                # Free GPU memory
                del output

            print("Training loss is " + str(total_loss))
            # Store the model corresponding to the least loss
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, "weights_epoch_" + str(epoch) + ".pt"))

            # Test model against the validation set.
            # Check the loss on the validation set, set to eval mode to ensure Dropout, batch norm behaves correctly
            self.eval()
            # Create data handler and data loader for validation set.
            total_dev_loss = 0
            # Ensure that gradients aren't computed since we don't need to back prop every other step is the same
            # as mentioned above
            with torch.no_grad():
                index = 0
                for i, batch in enumerate(dev_loader):
                    images = batch[0]
                    labels = batch[1].float()
                    if torch.cuda.is_available():
                        images = images.cuda()
                        labels = labels.cuda()

                    score = self.forward(images).squeeze(1)
                    output = loss(score, labels)

                    total_dev_loss += output.item()
                    del output
                    index = i
            print("Validation loss is " + str(total_dev_loss/index))

            # Store the model corresponding to the best f1 score
            if total_dev_loss < best_dev_loss:
                best_dev_loss = total_dev_loss
                torch.save(self.state_dict(), os.path.join(self.save_dir, "dev_weights_epoch_" + str(epoch) + ".pt"))

    # Predict the output label for the test set
    def predict(self, test_dir, batch_size):
        data_handler = TestDataHandler(test_dir)
        num_workers = 1
        loader = DataLoader(data_handler, batch_size, False, num_workers=num_workers, pin_memory=True)
        if torch.cuda.is_available():
            self.cuda()
        self.eval()
        predicted_labels = []
        ground_truth = []
        with torch.no_grad():
            try:
                for batch in loader:
                    images = batch[0]
                    names = batch[1]
                    if torch.cuda.is_available():
                        images = images.cuda()
                    speed = self.forward(images).squeeze(1)
                    predicted_labels.extend(list(speed.cpu().detach().numpy()))
            except Exception as e:
                print(e)
                pass
        with open("test.txt","w+") as f:
            for pred in predicted_labels:
                f.write(str(pred)+"\n")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action="store", default=0.0001, type=float,
                        help='The learning rate of the network')
    parser.add_argument('--batch_size', action='store', type=int, default=32,
                        help="The batch size for training.")
    parser.add_argument('--epochs', action='store', type=int, default=50, help="The number of epochs during train time")
    parser.add_argument('--momentum', action='store', type=float, default=0.9, help="The momentum for an optimizer")
    parser.add_argument('--width', action='store', type=int, default=66, help='Width of the images.')
    parser.add_argument('--height', action='store', type=int, default=220, help='Height of the images.')
    parser.add_argument('--save_dir', action='store', type=str, default='saved_weights',
                        help='Directory in which weights will be saved')
    parser.add_argument('--gt_file_path', action='store', type=str, default='data/train.txt',
                        help='Directory in which weights will be saved')
    parser.add_argument('--weights_path', action='store', type=str, default='saved_weights/dev_weights_epoch_4.pt',
                        help='Path of the weights to be loaded during predict time.')
    parser.add_argument('--train_dir', action='store', type=str, default="data/frames_train",
                        help='Directory containing the training images')
    parser.add_argument('--test_dir', action='store', type=str, default="data/frames_test",
                        help='Directroy containing the test images')
    parser.add_argument('--ground_truth_masks_dir', action='store', type=str, default='../test256/masks_pngs',
                        help='Directory containing the ground truth masks.')
    parser.add_argument('--output_dir', action='store', type=str, default="../results",
                        help='Directory that the output activation maps would be saved to')
    parser.add_argument('--mode', action='store', choices=['train', 'predict'], default='predict',
                        help='In train mode the network will be trained, in predict mode the network will use'
                             'the default weights to predict the pixel wise classes', required=False)

    args = parser.parse_args()

    obj = Module(width=args.width, height=args.height, save_dir=args.save_dir)
    if args.mode == 'train':
        obj.train_model(train_dir=args.train_dir, gt_file_path=args.gt_file_path, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs)
    elif args.mode == 'predict':
        obj.load_model(args.weights_path)
        obj.predict(args.test_dir,args.batch_size)