import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

def create_pd_df(input_f,label):
    inpf = np.genfromtxt(input_f,delimiter=',')
    tup_list = list(map(lambda x: [x,label],inpf))
    df = pd.DataFrame(tup_list, columns =['Data', 'Cancer'])
    return df  

luad = create_pd_df('luad_umap.csv','luad')
gbm = create_pd_df('gbm_umap.csv','gbm')
frames = [luad,gbm]
combined = pd.concat(frames)


reducer = umap.UMAP(n_neighbors=2,min_dist=.01,metric="red")
emb_input = np.array(list(combined["Data"].values))
scaled_emb_input = StandardScaler().fit_transform(emb_input)
embedding = reducer.fit_transform(scaled_emb_input)

plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in combined.Cancer.map({"gbm":0, "luad":1})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP Projection',fontsize=24)
plt.show()
