import pandas as pd
import os

dataset = "psp_plates_new_2"
data_dir = "/Users/Stanley/Desktop/Tyrrell Lab/ROP Project/PCA-Clustering-Project/data/"
paths = []
for root, dirs, files in os.walk(data_dir+dataset, topdown=False):
   for name in files:
      paths.append(os.path.join(root, name))
      
paths_series = pd.Series(paths)
features_idx = paths_series.str.contains("_features.csv")
features_paths = paths_series[features_idx].reset_index(drop=True)


def merge(x):
    features = pd.read_csv(x).loc[:, "IDs":]
    
    preds = pd.read_csv(x.replace("_features", "")).loc[:, "IDs":"Pred"]
    
    final = pd.merge(features, preds,
             how="left",
             on="IDs")
    try:
        final = final.rename(columns={"IDs": "img_ids",
                              "Pred": "predictions",
                              "Labels": "labels"
                              })
    except:
        pass
    
    final.to_csv(x.replace("_features", "_final"), index=False)

features_paths.map(merge)