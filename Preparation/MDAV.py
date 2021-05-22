import pandas as pd
import numpy as np

#Import Data
values = pd.read_csv("fraud.csv", usecols=["lat", "long"])
#Value of k provides k-anonymity
k = 10

#Normalise/Standardise data before distance measures
mean = values.mean()
stdev = values.std()
stand = (values - mean)/stdev
del(values)

#List of clusters 
k_list = []

#While more than 2k points remain in the data
while (len(stand) > 2*k):
    #Calculate centroid of the data and all the distances of points from that centroid
    centroid = stand.mean()
    distance = ((stand['lat'] - centroid["lat"]) ** 2 + (stand['long'] - centroid["long"]) ** 2) ** 0.5

    #Find r - the furthest point in the data from the centroid
    r = distance.idxmax()
    r_dis = ((stand['lat'] - stand["lat"][r]) ** 2 + (stand['long'] - stand["long"][r]) ** 2) ** 0.5

    #Find s - the furthest point in the data from r
    s = r_dis.idxmax()

    #Get the closest k points in the dataset around r and add them as a cluster to the list (then remove them from main dataset)
    kr = r_dis.nsmallest(k).index
    k_list.append(stand.loc[kr])
    stand = stand.drop(index=kr)
    
    #Get the closest k points in the dataset around s and add them as a cluster to the list (then remove them from main dataset)
    s_dis = ((stand['lat'] - stand["lat"][s]) ** 2 + (stand['long'] - stand["long"][s]) ** 2) ** 0.5
    ks = s_dis.nsmallest(k).index
    k_list.append(stand.loc[ks])
    stand = stand.drop(index=ks)
    
#For each of the clusters we set all values as the centroid of that cluster
for cluster in k_list:
    #Includes denormalisation of the values so they can be used in dataset
    centroid = (cluster.mean() * stdev) + mean
    cluster["lat"] = centroid["lat"]
    cluster["long"] = centroid["long"]

#If there are more than k values remaining in the dataset at end of loop you give them their own cluster
if (len(stand) > k):
    #Includes denormalisation of the values so they can be used in dataset
    centroid = (stand.mean() * stdev) + mean
    stand["lat"] = centroid["lat"]
    stand["long"] = centroid["long"]
    k_list.append(stand)
#Otherwise you simply add them to the closest cluster
else:
    #Yet to implement as it doesn't happen in our dataset
    print(stand)

#Put the updated location values back into the original dataset
replacement = pd.concat(k_list).sort_index()
data = pd.read_csv("fraud.csv", index_col=0)
data["lat"] = replacement["lat"]
data["long"] = replacement["long"]
data.to_csv("mdav.csv")