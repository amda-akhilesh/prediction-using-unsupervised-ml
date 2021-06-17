import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv",index_col= 0)

x = df.iloc[:,:4].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

pd.DataFrame({"Number of Clusters":range(1,11),"WCSS":wcss})

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid()
plt.show()


kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.figure(figsize=[10,8])
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],
            s =  100, c = "red", label = 'Iris_setosa')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],
            s =  100, c = "blue", label = 'Iris_versicolor')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],
            s =  100, c = "green", label = 'Iris_virginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()

