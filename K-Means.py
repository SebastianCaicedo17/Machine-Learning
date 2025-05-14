from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas
import numpy as np

df_notes = pandas.read_csv("notes.csv").drop(["Eleve"], axis=1)

####   Normalisation Standard ######

standard_scaler = StandardScaler()
normalized_problem = standard_scaler.fit_transform(df_notes)

## Split en Train test

x_train, x_test = train_test_split(normalized_problem)

## Elbow Méthode 
def Elbow_method (x_train):
    k_range = range(1,10)
    inertias =[] #Inertias = Variance
    for k in k_range:
        kmeans = KMeans(n_clusters = k )
        kmeans.fit(x_train)
        inertias.append(kmeans.inertia_)
    return k_range, inertias

def plot (k_range, inertias):
    plt.plot(k_range, inertias, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie (SSE)')
    plt.title('Méthode du coude')
    plt.grid(True)
    plt.show()

k_range, inertias = Elbow_method(x_train)
plot(k_range, inertias)
