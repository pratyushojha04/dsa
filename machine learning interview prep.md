1. Explain the Bias-Variance Tradeoff.
Details:

Bias refers to the error introduced by approximating a real-world problem with a simplified model. High bias indicates that the model is too simplistic, leading to systematic errors (underfitting).
Variance refers to the model’s sensitivity to fluctuations in the training data. High variance indicates that the model is too complex and overfits the training data, capturing noise rather than the underlying pattern.
Tradeoff: As you increase model complexity (e.g., adding more features or polynomial terms), bias decreases but variance increases. Conversely, a simpler model has higher bias and lower variance. The goal is to find a balance where both bias and variance are minimized, achieving a model that generalizes well to new data.
2. What is Cross-Validation and Why is it Important?
Details:

Cross-Validation is a technique used to assess how the results of a statistical analysis will generalize to an independent dataset. It is used to ensure that the model is robust and not overfitting to a particular subset of the data.
K-Fold Cross-Validation: The dataset is divided into 
𝑘
k subsets (folds). The model is trained on 
𝑘
−
1
k−1 folds and tested on the remaining fold. This process is repeated 
𝑘
k times, with each fold used exactly once as a test set. The final performance metric is the average of the metrics from each fold.
Importance: It helps in assessing the model’s performance more reliably compared to a single train-test split. It also ensures that every data point gets to be in both training and test sets.
3. How Does a Decision Tree Work?
Details:

Decision Tree: A decision tree is a model that splits the data into subsets based on the feature values, aiming to create a tree-like structure with decision nodes and leaf nodes.
Splits: At each node, the data is split based on a feature that provides the best separation of the target variable. Measures like Gini impurity or entropy (information gain) are used to evaluate the quality of splits.
Gini Impurity: Measures the probability of incorrectly classifying a randomly chosen element from the dataset. The formula is 
𝐺
𝑖
𝑛
𝑖
=
1
−
∑
(
𝑝
𝑖
2
)
Gini=1−∑(p 
i
2
​
 ), where 
𝑝
𝑖
p 
i
​
  is the probability of an element being classified into class 
𝑖
i.
Entropy: Measures the impurity or randomness. The formula is 
𝐸
𝑛
𝑡
𝑟
𝑜
𝑝
𝑦
=
−
∑
(
𝑝
𝑖
⋅
log
⁡
2
(
𝑝
𝑖
)
)
Entropy=−∑(p 
i
​
 ⋅log 
2
​
 (p 
i
​
 )).
Advantages: Easy to understand and interpret, handles both numerical and categorical data.
Disadvantages: Prone to overfitting, sensitive to noisy data.
4. Describe the Concept of Regularization in Machine Learning.
Details:

Regularization: Techniques used to prevent overfitting by adding a penalty to the loss function for large coefficients or complex models.
L1 Regularization (Lasso): Adds a penalty proportional to the absolute value of the coefficients. It can lead to sparse models where some coefficients are exactly zero, effectively performing feature selection.
L2 Regularization (Ridge): Adds a penalty proportional to the square of the coefficients. It helps in shrinking coefficients but doesn’t necessarily zero them out.
Importance: Regularization helps in improving model generalization by discouraging overly complex models that fit the noise in the training data.
5. What is the Difference Between Classification and Regression?
Details:

Classification: Involves predicting a categorical label. For example, predicting whether an email is spam or not spam. Common algorithms include Logistic Regression, Decision Trees, and Support Vector Machines. Evaluation metrics include accuracy, precision, recall, and F1 score.
Regression: Involves predicting a continuous value. For example, predicting house prices based on features like size and location. Common algorithms include Linear Regression, Polynomial Regression, and Support Vector Regression. Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
6. Explain the Working of a Support Vector Machine (SVM).
Details:

SVM: A supervised learning model that finds the optimal hyperplane which maximizes the margin between two classes in the feature space.
Hyperplane: A decision boundary that separates classes. The goal is to find a hyperplane that maximizes the distance (margin) between the closest points of the classes (support vectors).
Kernel Functions: For non-linearly separable data, SVM uses kernel functions (e.g., polynomial, radial basis function) to map data to a higher-dimensional space where a linear hyperplane can be used.
Support Vectors: Data points that lie closest to the hyperplane and are critical in defining the position and orientation of the hyperplane.
7. What Are Hyperparameters and How Do You Tune Them?
Details:

Hyperparameters: Parameters that are set before the learning process begins and control the learning process itself (e.g., learning rate, number of trees in a Random Forest, etc.).
Tuning Methods:
Grid Search: Exhaustively searches through a specified subset of hyperparameters. This can be computationally expensive but thorough.
Random Search: Samples a random subset of hyperparameters. It can be more efficient than grid search, especially when some hyperparameters are more important than others.
Bayesian Optimization: Uses probabilistic models to find the optimal hyperparameters more efficiently by considering past evaluation results.
8. What is Principal Component Analysis (PCA) and How is it Used?
Details:

PCA: A dimensionality reduction technique that transforms the data into a set of orthogonal (uncorrelated) components that capture the most variance in the data.
Process:
Standardization: Scale the data to have zero mean and unit variance.
Covariance Matrix: Compute the covariance matrix of the data.
Eigen Decomposition: Perform eigenvalue decomposition to find eigenvectors (principal components) and eigenvalues (variance captured).
Projection: Project the data onto the principal components to reduce dimensionality while preserving as much variance as possible.
Usage: PCA is used for reducing the number of features, visualizing high-dimensional data, and improving the efficiency of machine learning algorithms.
9. Describe the Concept of Gradient Descent.
Details:

Gradient Descent: An optimization algorithm used to minimize the loss function by iteratively updating model parameters in the direction of the steepest descent.
Process:
Compute Gradient: Calculate the gradient (partial derivatives) of the loss function with respect to each parameter.
Update Parameters: Adjust the parameters by moving them in the direction opposite to the gradient. The step size is determined by the learning rate.
Variants:
Batch Gradient Descent: Uses the entire dataset to compute the gradient and update parameters.
Stochastic Gradient Descent (SGD): Uses one data point at a time to compute the gradient, which makes it faster but more noisy.
Mini-Batch Gradient Descent: Uses a small random subset of data to compute the gradient, balancing between efficiency and accuracy.
10. What is Ensemble Learning and What Are Some Common Techniques?
Details:

Ensemble Learning: A technique that combines multiple models to improve overall performance by leveraging the strengths of each model and reducing the impact of individual weaknesses.
Techniques:
Bagging: (Bootstrap Aggregating) Involves training multiple models on different subsets of the data and averaging their predictions. Example: Random Forests.
Boosting: Sequentially trains models, with each new model focusing on the errors made by the previous ones. Example: Gradient Boosting Machines (GBM), AdaBoost.
Stacking: Combines multiple models (base learners) and uses another model (meta-learner) to make the final prediction based on the predictions of base learners.
