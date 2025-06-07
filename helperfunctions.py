#File: helperfunctions.py
# This file contains a variety of functions (simple string processing; SQLite helper functions, graphing
# functions etc) which are used in other files. 

# Author: Elizabeth Javor elizabethjavor@proton.me
# May 2025

###################################################################


import matplotlib.pyplot as plt
import torch

# quote surrounds with single quotes (we need to concatenate "'Alabama'" and not "Alabama" to a string so the database is queried with the correct type)
def quote(quotation):
    return "'" + quotation + "'"

# Takes a string and makes it an acceptable SQLite identifier by changing illegal characters
def sqlify(string):
    newstring = string.strip()
    if newstring and newstring[0].isdigit():
        newstring = "num" + newstring
    for char in newstring:
        if not (char.isdigit() or char.isalpha() or char == "_" or char == "$"):
            newstring = newstring.replace(char, "_")
    return newstring

# cursor.execute returns information in a tuple, even if it is a tuple with one element
# This function extracts the element from the tuple, or if it is a list of one-element tuples, to simply a list
# Query is the query to execute
def fetchfirst(cursor, query):
    result = cursor.execute(query).fetchall()
    if len(result) == 0:
        #I never (try to) fetch something with a result of 0; this mean's something's wrong.
        raise ValueError("fetchfirst returned None. Did you expect it to return something?")
    if len(result) == 1:
        return result[0][0]
    else:
        newlist = []
        for item in result:
            newlist.append(item[0])
        return newlist

#Prints the proportion of each statistic that is null 
def nonechecker(statlist, cause, cur):
    print(cause)
    total = fetchfirst(cur, f"SELECT COUNT(*) FROM {sqlify(cause)}")
    for i in range(len(statlist)):
        print(f"Stat {i} Proportion Null:")
        print(fetchfirst(cur, f"SELECT COUNT(*) FROM {sqlify(cause)} WHERE Stat{i} IS NULL")/total)

# Checks if table bestparams exists
#Creates it and prefills with presets if it does not

def paramexist(data):
    exists = fetchfirst(data.cur, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='bestparams'")
    if exists == 0:
        data.cur.executescript("""
        CREATE TABLE bestparams (features INT, cause TEXT, model TEXT, split FLOAT, epochs INT,  lr FLOAT);
        INSERT INTO bestparams (features, cause, model, split, epochs, lr) VALUES (123456, 'diabetes', 'linear', 0.75, 1000, 0.01);
        INSERT INTO bestparams (features, cause, model, split, epochs, lr) VALUES (123456, 'diabetes', 'nn', 0.8, 2000, 0.01);
        INSERT INTO bestparams (features, cause, model, split, epochs, lr) VALUES (123456, 'heart_disease', 'linear', 0.75, 500, 0.01);
        INSERT INTO bestparams (features, cause, model, split, epochs, lr) VALUES (123456, 'heart_disease', 'nn', 0.75, 2000, 0.01);
                               """)

# Saves optimal gridsearch parameters to table bestparams
# Will not overwrite if there is already a parameter saved (just keeps first params)

def paramsave(data, modeltype, best):
    paramexist(data)
    feats = int("".join(str(i) for i in data.feats))
    
    exists = data.cur.execute("SELECT 1 FROM bestparams WHERE features = ? AND cause = ? AND model = ?", (feats,sqlify(data.name), modeltype))
    if not exists.fetchone():
        
        data.cur.execute("INSERT INTO bestparams (features, cause, model, split, epochs, lr) VALUES (?, ?, ?, ?, ?, ?)", 
                    (feats, sqlify(data.name), modeltype, best[0], best[1], best[2]))
        
        data.con.commit()

#Gets optimal previously gridsearched parameters from the table. 
#If there are no optimal parameters presaved, then just uses the parameters for all features (preloaded)
def paramfetch(data, modeltype):
    paramexist(data)
    feats = int("".join(str(i) for i in data.feats))
    exists = data.cur.execute(
        "SELECT split, epochs, lr FROM bestparams WHERE features = ? AND cause = ? AND model = ?",
        (feats, sqlify(data.name), modeltype)).fetchone()

    if exists:
        return exists
    else:
        print("No presaved hyperparameters found for this model. Model with all features will be used.")
        return data.cur.execute(
            "SELECT split, epochs, lr FROM bestparams WHERE features = 123456 AND cause = ? AND model = ?",
            (sqlify(data.name), modeltype)).fetchone()

#Makes graph2 (activated by -g2 flag)


def predictgraphmaker(ds, model):
    x = []
    ytrue = []
    ypred = []
    for i in range(len(ds)):
        xraw, yraw = ds[i]
        #normalize x to be fed into the model
        xnorm = (xraw - ds.xmean) / (ds.xstd + 1e-6)
        y = model(xnorm.unsqueeze(0)).item()
        yunnorm = y * ds.ystd + ds.ymean
        #unnormalize y after it's been inputted from the model
        x.append(xraw)
        ytrue.append(yraw.item())
        ypred.append(yunnorm)

    plt.scatter(ytrue, ypred, alpha=0.7)
    plt.xlabel("Actual Deaths per Million")
    plt.ylabel("Predicted Deaths per Million")
    plt.title(f"Predicted vs Actual DPM ({ds.name.title()}) with features {",".join(str(i) for i in ds.feats)}")
    plt.plot([min(ytrue), max(ytrue)], [min(ytrue), max(ytrue)], color="red")
    plt.grid(True)
    plt.show()

#Creates graph1 (activated by -g1 flag); graph of each statistic vs. deaths per million from the given cause
# as a scatterplot; if given a model, displays a red line of model predictions

def scattergraphmaker(ds, model = None):
    ydata = fetchfirst(ds.cur, f"SELECT Deaths_Per_Million FROM {ds.table}")
    abrstat = [
        "Overweight",
        "Obese",
        "No Physical Activity",
        "Muscle",
        "Short Aerobic",
        "Short Aerobic and Muscle",
        "Long Aerobic",
        "No fruit",
        "No vegetable",
    ]
    if model is not None:
        figure, axis = plt.subplots(1, len(ds.feats), constrained_layout=True)
        for i, feat in enumerate(ds.feats):
            #If there is only a singular element, axis is not subscriptable; the below if-else fixes this.
            if len(ds.feats) == 1:
                axissub = axis
            else:
                axissub = axis[i]
            axissub.scatter(
                fetchfirst(ds.cur, f"SELECT Stat{feat} FROM {ds.table}"), ydata
            )
            xmin, xmax = axissub.get_xlim() #so line stretches entire graph
            x_vals = torch.linspace(xmin, xmax, steps=100) 
            x_full = torch.zeros(100, len(ds.feats))  
            x_full[:, i] = (x_vals - ds.xmean[i]) / (ds.xstd[i] + 1e-6) 
            #adds .000001 to prevent div by - errors. just normalization of x for feeding into model
            y_vals = model(x_full).squeeze()
            y_vals = y_vals * ds.ystd + ds.ymean #Unnormalizes y
            axissub.plot(x_vals, y_vals.detach(), color='red')
            axissub.set_xlabel(abrstat[feat], size="small")
            axissub.set_ylabel(f"DPM from {ds.name.title()}", size="small")
    #With no model given, simply graphs scatterplots of all data, in a 3 x 3 grid. 
    if model is None:
        figure, axis = plt.subplots(3, 3)
        for i, stat in enumerate(abrstat):
            axis[i%3,i//3].scatter(
            fetchfirst(ds.cur, f"SELECT Stat{i} FROM {ds.table}"), ydata
         )
            axis[i%3,i//3].set_xlabel(abrstat[i], size="small")
            axis[i%3,i//3].set_ylabel(f"DPM from {ds.name.title()}", size="small")
    plt.suptitle(
        f"Various Statewide Statistics vs. Deaths per Million from {ds.name.title()} (Across years 2011 - 2017)",
        size="large")
    plt.show()
