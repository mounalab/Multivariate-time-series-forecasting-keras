# Siamese Neural Network with Keras

This project provides a Siamese neural network implementation with Keras framework.


In the example,
1. We simply use a multi-layer Perceptron as the sub-network that generates the feature embeddings (encoding)
2. We used a Euclidean distance to measure the similarity between the two output embeddings. In other words, our Siamese network is trying to learn an embedding function that maps feature vectors to a feature space where Euclidean distance between embeddings reflect the semantic similarity between features  
3. We use the constrastive loss as loss function for the training of the Siamese network [1]

The general form [1] of contrastive loss function $\mathcal{L}(W)$ which is defined as :

\begin{equation}
%\begin{split}
\mathcal{L}(W) = \sum_{i=1}^{P} L(W, (Y, X_1, X_2)^i)
%\end{split}
\end{equation}
\begin{equation}
%\begin{split}
L(W, (Y, X_1, X_2)^i) = (1 - Y) L_S(D_W^i) + Y L_D(D_W^i)
%\end{split}
\end{equation}

where $(Y, X_1, X_2)^i$ is the $i$th labeled sample pair, $P$ is the number of training pairs, $L_S$ is the partial loss function for a pair of similar data points, and $L_D$ is the partial loss function of a pair of dissimilar data points. 

$L_S$ and $L_D$ are defined so that $D_W$ has low values for similar inputs and high values for dissimilar inputs :

\begin{equation}
%\begin{split}
L_S(W, X_1, X_2) = \frac{1}{2} (D_W)2 
%\end{split}
\end{equation}

\begin{equation}
%\begin{split}
L_D(W, X_1, X_2) = \frac{1}{2} {max(0 , m - D_W)}^2
%\end{split}
\end{equation}

where $m>0$ is a margin that is used to hold constraint, i.e. when two inputs are dissimilar, and so the distance between them is bigger than a margin, they do not contribute to the loss. This allows no wasted computations for enlarging distance between embeddings of dissimilar inputs when these are distant enough. The contrastive term $L_D$ avoids the loss made zero by simply setting embeddings $G_W$ to a constant.



## References

[1] Hadsell, R., Chopra, S., & LeCun, Y. (2006, June). Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06) (Vol. 2, pp. 1735-1742). IEEE.
