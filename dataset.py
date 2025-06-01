# File: dataset.py
# File containing the custom Pytorch dataset which uses SQLite to query data organized in preprocessing.

# Author: Elizabeth Javor elizabethjavor@proton.me
# May 2025

###################################################################


import sqlite3
import torch
import os
from torch.utils.data import Dataset
from preprocessing import main as createhealthdb
from helperfunctions import *

deathpos = 3 #This is where the deaths-per-million statistic is stored (3rd column of SQL table)
statdelay = 4 #This is the column number of the "Stat0" column


class sqlhealthds(Dataset):
    def __init__(self, table, feats):
        if not os.path.exists("health.db"):
            createhealthdb()
        self.con = sqlite3.connect("health.db")
        self.cur = self.con.cursor()
        strng = strng = " AND ".join(f"Stat{i} IS NOT NULL" for i in feats)
        tabstr = f"Non_null_Stats{'_'.join(str(i) for i in feats)}"
        sqledtab= sqlify(table)
        nonnull = fetchfirst(self.cur, f"SELECT COUNT (*) FROM {sqledtab} WHERE {strng}")
        #In the case where some data is NULL, select only rows with nonnull data and creates a table 
        #of those rows to use
        #This happens for Stat 4 and above
        if nonnull < fetchfirst(self.cur, f"SELECT COUNT (*) FROM {sqledtab}"):
            self.cur.execute(f"DROP TABLE IF EXISTS {tabstr}")
            self.cur.execute(f"CREATE TABLE {tabstr} AS SELECT * FROM {sqledtab} WHERE {strng}")
            self.cur.execute(f"UPDATE {tabstr} SET id = rowid")
            self.table = tabstr
            self.con.commit()
        else:
            self.table = sqlify(table)
        self.name = table
        self.feats  = feats
        
#Returns the length of the table being used, as decided/defined in __init__. 
    def __len__(self):
        return fetchfirst(self.cur, f"SELECT COUNT (*) FROM {self.table}")
#Returns a (nonnull) normalized item, from id number in the data set
#Can also reference (for my own sanity checks) data by state name and year
    def __getitem__(self, iden):
        if isinstance(iden, int):
            row = self.cur.execute(f"SELECT * FROM {self.table} WHERE id = ?", (iden + 1,) ).fetchone()
        elif isinstance(iden, tuple) and len(iden) == 2:
            row = self.cur.execute(f"SELECT * FROM {self.table} WHERE State = ? AND Year = ?", (quote(iden[0]),iden[1])).fetchone()
        else:
            raise TypeError("Invalid index type. Must be int or (state, year) tuple.")
        y =torch.tensor(row[deathpos],dtype=torch.float32)
        x = torch.FloatTensor([row[i+statdelay] for i in self.feats])
        return x,y
#Below function splits the data (into training/validation) and normalizes it.
#Calculates and stores (as class attributes) the mean and std of the training data
#for both x (each feature, the stats) and y (the "labels", deaths per million from the cause)
    def splitandnorm(self, split):
        trainsize = int(split*len(self))
        valsize = len(self) - trainsize 
        traindata, valdata = torch.utils.data.random_split(self, [trainsize, valsize],generator=torch.Generator().manual_seed(525))
        #Since splitting turned them into datasets, we have to stack back into a tensor
        trainx = torch.stack([traindata[i][0] for i in range(trainsize)])
        trainy = torch.stack([traindata[i][1] for i in range(trainsize)])
        xmean = trainx.mean(dim=0)
        xstd = trainx.std(dim=0)
        ymean = trainy.mean()
        ystd = trainy.std()
        valx = torch.stack([valdata[i][0] for i in range(valsize)])
        valy = torch.stack([valdata[i][1] for i in range(valsize)])
        normtrainx = (trainx - xmean)/(xstd + 1e-6)
        normvalx = (valx - xmean)/(xstd+ 1e-6)
        normtrainy = (trainy - ymean)/(ystd + 1e-6)
        normvaly = (valy - ymean)/(ystd+ 1e-6)
        #Storage of mean and std for training data
        self.xmean = xmean
        self.xstd = xstd
        self.ymean = ymean
        self.ystd = ystd
        return torch.utils.data.TensorDataset(normtrainx, normtrainy), torch.utils.data.TensorDataset(normvalx, normvaly)
    def close(self):
        self.con.close()

    

    

