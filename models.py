#File: models.py
#File containing the command line interface (main file for running code), and the neural network
#and linear regression models (and functions for training, validation, and grid search).

# Author: Elizabeth Javor elizabethjavor@proton.me
# May 2025

###################################################################


import torch
import matplotlib.pyplot as plt
import logging
import os
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
    if not os.path.exists("logs"):
        os.makedirs("logs")
    filename = f"logs/{data.name}_{modeltype}_Batch32AdamMSE_FEATS_{feat_str}.log"
    logging.basicConfig(filename=filename, format='%(message)s',filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    minloss = torch.finfo(torch.float).max
    infodict = {}
    for i in range(trials):
        for epoch in epochs:
            for split in splits:
                for lr in lrs:
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
                        mindict = model.state_dict
        logger.info(f"Smallest Loss achieved trial {i}: {minloss}, with {mintuple[0]} split, {mintuple[1]} epochs, {mintuple[2]} learning rate")
    for tuple in infodict.keys():
        infodict[tuple] /= trials 
        logger.info(f"Average loss for {tuple} over {trials} trials: {infodict[tuple]}")
    best = min(infodict.items(), key=lambda x: x[1])[0] #sorts by loss, returns 1st place
    logger.info(f"Best model on average had loss {infodict[best]} with {best[0]} split, {best[1]} epochs, {best[2]} learning rate")
    #Below saves weights & biases to the log
    if modeltype == "linear":
        logger.info(f"Weight data: {model.weight.data}")
        logger.info(f"Bias data: {model.bias.data}")
    #Weights are only saved for linear because they are simple and understandable for linear; too much data for nn
    return best

#Parses command line arguments and runs desired commands

#Usage: models.py <cause> <modeltype> [--features 1 2 3 4 5] [--gs N] [-g]
#See README for more detailed usage instructions

def main():
    

    parser = argparse.ArgumentParser(description="Train a model on public health data.")
    parser.add_argument("cause", type=str.lower, help="Cause of death to predict (e.g., Diabetes, Heart disease)")
    parser.add_argument("modeltype", type=str.lower, choices=["linear", "nn"], help="Type of model: Linear or NN")
    # Optional arguments for features, graphing, or gridsearch
    parser.add_argument("--features", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                        help="List of feature indices to use (default: 1 2 3 4 5)")
    parser.add_argument("--gs", type=int, default=0, help="Number of trials for grid search. 0 (default) does not run gridsearch, and uses hyperparameters from best previous gridsearch (preprogrammed).")
    parser.add_argument("-g1", "--graph1", action="store_true",help="If set, display graphs of the model predictions vs each feature (1 graph each feature).")
    parser.add_argument("-g2", "--graph2", action="store_true",help="If set, display graphs of the model predictions vs true values.")
    parser.add_argument("-s", "--save", action="store_true",help="If set, saves the model (or best model from gridsearch)."
    )
    args = parser.parse_args()
    data = sqlhealthds(args.cause, args.features)

    #If grid search is on, grid searches for num of trials given and saves data

    if args.gs != 0:
        best = gridsearch(data, args.modeltype, args.gs)
        paramsave(data, args.modeltype, best)

    else:
        try:
            best = paramfetch(data, args.modeltype)
        except KeyError:
            #In theory, can test with other causes of death (not tested) which would not have any defaults.
            raise ValueError(f"No default parameters found for {args.cause} with {args.modeltype}. Please run gridsearch.")
    if args.modeltype == "linear":
        finalmodel = torch.nn.Linear(len(data.feats), 1)
    else:
        finalmodel = NeuralNetwork(len(data.feats))

    finalmodel, finalloss = trainandvalidate(data, finalmodel, best[0], best[1], best[2])

   #Prints data about the model if gridsearch isn't used (if it is, this data is logged)
    if args.gs == 0:
        print(f"Model validation loss: {finalloss}")


    if args.graph1:
        scattergraphmaker(data, finalmodel)
    if args.graph2:
        predictgraphmaker(data, finalmodel)
    
    #Model is saved in "models" folder if -s flag is used
    if args.save:
        if not os.path.exists("models"):
            os.makedirs("models")
        filename = f"models/{data.name}_{args.modeltype}_Batch32AdamMSE_FEATS_{args.features}"
        i = 1
        newfilename = filename
        while True:
            if not os.path.isfile(newfilename):
                break
            else:
                newfilename = filename + str(i) + ".pt"
                i += 1
        newfilename = newfilename + ".pt"
        torch.save(finalmodel.state_dict(), newfilename)
                   

    data.close()
if __name__ == "__main__":
    main()