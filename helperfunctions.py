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


#Creates graphs of each statistic vs. deaths per million from the given cause, and displays them
#If given a model, displays model predictions as a red line

def graphmaker(ds, model = None):
    ydata = fetchfirst(ds.cur, f"SELECT Deaths_Per_Million FROM {sqlify(ds.table)}")
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
            axissub.set_ylabel(f"DPM from {ds.name.capitalize()}", size="small")
    #With no model given, simply graphs scatterplots of all data, in a 3 x 3 grid. 
    if model is None:
        figure, axis = plt.subplots(3, 3)
        for i, stat in enumerate(abrstat):
            axis[i%3,i//3].scatter(
            fetchfirst(ds.cur, f"SELECT Stat{i} FROM {ds.name}"), ydata
         )
            axis[i%3,i//3].set_xlabel(abrstat[i], size="small")
            axis[i%3,i//3].set_ylabel(f"DPM from {ds.name}", size="small")
    plt.suptitle(
        f"Various Statewide Statistics vs. Deaths per Million from {ds.name.capitalize()} (Across years 2011 - 2017)",
        size="large")
    plt.show()
