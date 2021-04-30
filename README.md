# Understanding the Impact of Dataset Heterogeneity on Convolutional Neural Network Model Training

Proposed by [Mendez, Calderon and Tyrrell](https://link.springer.com/chapter/10.1007/978-3-030-41005-6_21) in 2019. The methodology is as follows:

1. **Feature Extraction**

> Before the prediction layer in the CNN model, neural network activations are extracted from images from the *training* set and *testing* set.

2. **Dimensionality Reduction**

> Principal Component Analysis is used to transform the high-dimensional features. The top principal components are selected, and the rest are dropped. 
> Principal components (PC) are found for the training set, while both the training and testing set features are projected onto these PCs.

3. **Cluster Analysis**

> Cluster the transformed training set. Use found cluster centers for assigning the transformed testing set into clusters.
> For each cluster assigned in the testing set, calculate a mean CNN performance metric. (e.g. accuracy for classification, mean squared error (MSE) for regression)

**Coefficient of Variation** (CV) of clusters' mean CNN model performances is used to measure how model training is affected by dataset heterogeneity.

*NOTE: CV is a standardized measure of spread.*

---
## Dimensionality Reduction

Here, we answer the following questions:

#### <ins>Does the number of principal components affect our ability to assess the impact of dataset heterogeneity? </ins>

Yes. However, this varies by sample size and dataset.

![alt text](https://github.com/stan-hua/pca-clustering/blob/main/results/graphs/presentation_graphs/cv%20vs.%20num_pcs%20(PSP%20Plates).png)


#### <ins>How many principal components do we include?</ins>

We suggest ***Minimum Mode CV*** for selecting the number of principal components. But how?
1. Iterate cluster analysis from 1 to **n** PCs
2. Select the minimum number of principal components that yields the most frequently appearing value of CV



---
## Possible Questions
#### <ins>When is a dataset heterogeneous?</ins>
It is common when samples of a dataset differ greatly from one another. This can happen due to noise or from the nature of the data.

#### <ins>Why does the heterogeneity of a dataset matter?</ins>
When the size of the dataset is small (relative), deep learning models may have trouble learning how to perform a dataset-related task (e.g. classification).

#### <ins>What is a Convolutional Neural Network (CNN)?</ins>
CNNs are a subclass of deep learning models dealing with images. They are often trained to recognize objects in an image (object classification). To do so, CNNs are thought to learn how to represent images in a way that assists with a specific task.

&nbsp;

In fulfillment of the Research Opportunities Program at the University of Toronto.
