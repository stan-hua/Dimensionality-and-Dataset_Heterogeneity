# Directory Structure
    .
    ├── data                    # CNN-extracted features
    ├── results                 
    │   ├── graphs              # Visualizations of experiments
    │   ├── dataset             # Resulting CV from iteration
    │   └── pc_selection        # Number of PCs suggested by Min. Mode CV and Cumulative Percent Variance
    └── scripts                 
        ├── main.py             # Used to perform iterative clustering
        ├── pca.py              # Subclass of sklearn PCA
        └── clustering.py       # Clustering class. Contains sklearn KMeans and skfuzz Fuzzy CMeans
        └── svcca.py            # Method for comparing NN representations
