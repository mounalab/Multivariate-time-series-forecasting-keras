# Siamese Neural Network with Keras

This project provides a Siamese neural network implementation with Keras framework.


In this use case,
1. We simply use a multi-layer Perceptron as the sub-network that generates the feature embeddings (encoding)
2. We used a Euclidean distance to measure the similarity between the two output embeddings. In other words, our Siamese network is trying to learn an embedding function that maps feature vectors to a feature space where Euclidean distance between embeddings reflect the semantic similarity between features  
3. e use the constrastive loss as loss function for the training of the Siamese network [1]


## References

[1] Hadsell, R., Chopra, S., & LeCun, Y. (2006, June). Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06) (Vol. 2, pp. 1735-1742). IEEE.
