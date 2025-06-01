#File: models.py
#File containing the command line interface (main file for running code), and the neural network
#and linear regression models (and functions for training, validation, and grid search).

# Author: Elizabeth Javor elizabethjavor@proton.me
# May 2025

###################################################################


import torch
import matplotlib.pyplot as plt
import logging
import argparse
from torch import nn
from torch.utils.data import DataLoader
from helperfunctions import *
from dataset import sqlhealthds

#A basic 3 layer network 

class NeuralNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nnops = nn.Sequential(
            nn.Linear(size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.nnops(self.flatten(x))

#Main training loop. 
#Takes in data and hyperparameters, splits data into training and validation,
#trains the model for the given number of epochs. Returns the model itself and its 
#minimum validation loss
#uses Adam optimizer and mean squared error loss, and a batch size of 32

def trainandvalidate(data, model, split, numepochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = torch.nn.MSELoss()
    traindata, valdata = data.splitandnorm(split)
    trainloader = DataLoader(traindata, batch_size=32, shuffle=True)
    valloader = DataLoader(valdata, batch_size=32)
    minval = torch.finfo(torch.float).max
    for epoch in range(numepochs):
        for xbatch, ybatch in trainloader:
            ybatch = ybatch.view(-1, 1)
            pred = model(xbatch)
            loss = criterion(pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval() #takes it off of training mode to do validation
        totvalloss = 0 
        with torch.no_grad():
            for xbatch, ybatch in valloader:
                ybatch = ybatch.view(-1, 1)
                pred = model(xbatch)
                valloss = criterion(pred, ybatch)
                totvalloss += valloss
            avgval = totvalloss/len(valloader)
            if avgval < minval:
                minval = avgval
        model.train()
    return model, minval

#Performs basic gridsearch with 3 built-in lists with options for number of epochs, training validation
#split, and learning rates (9 options in total). Takes in number of trials to perform; grid search
#is then averaged over all trials. Outputs information on the search in a log file called
#[data][type of model]Batch32AdamMSE_FEATS[all features trained on].log

def gridsearch(data, modeltype, trials):

    epochs = [500,1000,2000]
    splits = [.75, .8, .9]
    lrs = [.01,.001,.0001]
    feat_str = "_".join(map(str, data.feats))
    filename = f"{data.name}_{modeltype}_Batch32AdamMSE_FEATS_{feat_str}.log"
    logging.basicConfig(filename=filename, format='%(message)s',filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    minloss = torch.finfo(torch.float).max
    mintuple = ()
    infodict = {}
    for i in range(trials):
        #Uncomment the print lines to see gridsearch working
        #(I find comfort in seeing the print statements to prove the code is actually running)
        #print("trial")
        #print(i)
        for epoch in epochs:
            #print(epoch)
            for split in splits:
                #print(split)
                for lr in lrs:
                    #print(lr)
                    #below restarts model 
                    if modeltype == "linear":
                        model = torch.nn.Linear(len(data.feats),1)
                    elif modeltype == "nn":
                        model = NeuralNetwork(len(data.feats))
                    else:
                        raise ValueError(f"Invalid model type '{modeltype}'. Expected 'Linear' or 'NN'.")
                    model, loss = trainandvalidate(data, model, split, epoch, lr)
                    if i == 0:
                        infodict[(split, epoch, lr)] = loss
                    else:
                        infodict[(split, epoch, lr)] += loss
                    if loss<minloss:
                        minloss = loss
                        mintuple = (split, epoch, lr)
        logger.info(f"Smallest Loss achieved trial {i}: {minloss}, with {mintuple[0]} split, {mintuple[1]} epochs, {lr} learning rate")
    for tuple in infodict.keys():
        infodict[tuple] /= trials 
        logger.info(f"Average loss for {tuple} over {trials} trials: {infodict[tuple]}")
    best = min(infodict.items(), key=lambda x: x[1])[0] #sorts by loss, returns 1st place
    logger.info(f"Best model on average had loss {infodict[best]} with {mintuple[0]} split, {mintuple[1]} epochs, {lr} learning rate")
    return best

#Parses command line arguments and runs desired commands

#Usage: models.py <cause> <modeltype> [--features 1 2 3 4 5] [--gs N] [-g]
#See README for more detailed usage instructions

def main():
    #Bestdict holds best parameters to be used when grid search isn't run
    #Filled from previous gridsearches (can be seen in examplelogs)
    bestdict = {}
    bestdict[("diabetes", "linear")] = (.8, 1000, .0001)
    bestdict[("diabetes", "nn")] = (.8, 2000, .0001)
    bestdict[("heart disease", "linear")] = (.75, 500, .0001)
    bestdict[("heart disease", "nn")] = (.8, 2000, .0001)

    parser = argparse.ArgumentParser(description="Train a model on public health data.")
    parser.add_argument("cause", type=str.lower, help="Cause of death to predict (e.g., Diabetes, Heart disease)")
    parser.add_argument("modeltype", type=str.lower, choices=["linear", "nn"], help="Type of model: Linear or NN")
    # Optional arguments for features, graphing, or gridsearch
    parser.add_argument("--features", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of feature indices to use (default: 1 2 3 4 5)")
    parser.add_argument("--gs", type=int, default=0, help="Number of trials for grid search. 0 (default) does not run gridsearch, and uses hyperparameters from best previous gridsearch (preprogrammed).")
    parser.add_argument(
    "-g", "--graph",
    action="store_true",
    help="If set, display graphs of the model predictions."
    )
    args = parser.parse_args()
    data = sqlhealthds(args.cause, args.features)

    if args.gs != 0:
        best = gridsearch(data, args.modeltype, args.gs)

    else:
        try:
            best = bestdict[(args.cause, args.modeltype)]
        except KeyError:
            raise ValueError(f"No default parameters found for {args.cause} with {args.modeltype}. Please run gridsearch.")

    if args.modeltype == "linear":
        finalmodel = torch.nn.Linear(len(data.feats), 1)
    else:
        finalmodel = NeuralNetwork(len(data.feats))

    finalmodel, finalloss = trainandvalidate(data, finalmodel, best[0], best[1], best[2])
    print(f"Model validation loss: {finalloss}")

    if args.graph:
        graphmaker(data, finalmodel)
    
    data.close()
if __name__ == "__main__":
    main()