### Write Up

##### How and what I Learnt

I started in Aug by going through all the resources provided in the first mail. I may have forgotten the specifics by now but I got a good feel for the basic aspects of statistical learning, some of the vocabulary and metrics, and developed an interest in the field.

I chose decision trees for the task as it felt the most accessible and intuitive, even though I had spent some term learning about neural networks before that.

Before going in the research papers and the book, I watched a few yt videos to understand the basic idea, and then ended up relying on chatgpt to help me read the theory. 

For writing the code, copilot and chatgpt were a big help as i told them to guide me through the process step by step, explaining how popular code implementations work, incrementally building up the tree. After I got nice results w/ the very simple iris dataset, I decided I wanted to run it on random real world data, as that may lead to more interesting outcomes. Hence I took an IMDB dataset, and predicting the genre of a movie based on numbers like runtime and metascore should be like predicting something totally random, yet I was surprised when the tree and the forest got test accuracies around 0.6.

I definitely am not good at using Git and realized that I wasn't on the main branch for half the work I did. Anyhow I managed to make it work and show meaningful commits. I also ended up doing most of the work on a single file, but saved results from different experiments in the form of many screenshots. I also referred to the Stanford lecture for the theory.

I did not implement regression and categorical variables mainly because I wanted to get till making forests before making changes in the tree code that would take me time to properly understand.

From what I understand, a tree for regression is supposed to replace Gini impurity with a measure of numerical variation such as mean squared error, and store the mean target value at each leaf instead of the majority class. A regression forest would then average the predictions of its trees rather than using the majority. 
Categorical predictor variables could be handled by one-hot encoding (basically treating each category as a seperate feature with value 1 or 0) them into binary columns, allowing the existing numerical split logic to use thresholds such as 0.5. I've hear more advanced implementation could split directly on groups of categories, but I could not read much and understand it properly.

I couldnt get sklearn to run on my operating system as there was some problem (DLL - idk what it means) when it tried installing sci py. Hence I had to run it on google colab and was surprised when one of my model performed better than sklearn.

I also wanted to run it on a huge dataset which theoretically should be predictable - so the 56000 x 24 lifestyle one seemed like a great option. It also made me realize the insane amount of compute power required for training. This set was so big that I simply couldn't finish training it. I had to significantly shorten it(by more than 95%) to reduce the training time for a single tree to just over an hour on my laptop. (Also made copilot add a progress check to confirm its still running). I got a max accuracy of 20% on that data after trying atleast 8 to 9 different parameter sets and even a forest with mutually exclusive tree datasets, with some tests even giving an accuracy worse than a random guess. Guess job type cant be predicted based on factors like bmi, stress levels or blood sugar, or the model works horribly on this particular dataset. 

The fundamental question of whether a decision tree is actually useful today compared to other methods came to me quite late, after having a discussion with my friends about the task options we had. It's quite a traditional/classical algorithm, which although is used in some parts of industry, has largely been under the shadow of modern 'deep learning' algorithms like neural networks which can be used in 'non standard' areas like computer vision.

##### How Decision Trees and Forests Work

The heart of the decision tree is segmenting the entire observation space, traditionally orthogonal, and predicting the value of a point based on the class of the segment or leaf it lies within. Since we go top down, making simple 'greater than / less than' splits across single features at a time, the algorithm is relatively fast to train and use. The optimum split is decided at that node based on what results in the least information loss, or maximum information gain, depending on the impurity measure used, such as Gini impurity or entropy. Most functions to measure information or impurity form a curve like the one shown in the image below.

Decision trees are fundamentally high-variance models. They are extremely sensitive to the specifics of the training data, and early splits have an outsized influence on later ones. Factors like limiting the maximum depth, limiting the leaf size, or stopping splits when the node size is too small can reduce variance and hence reduce overfitting. Another method is pruning, which involves growing out a full tree and then trimming the less useful branches.

The whole point of using ensembles (for example, a random forest) is to reduce variance.

###### Bagging

Bagging, short for bootstrap aggregating, is basically training a bunch of trees on different random samples of the same dataset and then combining their predictions. To make each sample, we draw rows from the training data with replacement - repetitions allowed (only around 63% points end up unique), so some observations appear more than once while others are left out. That means each tree sees a slightly different version of the data, so they are not all making the same mistakes.

At prediction time, we aggregate the outputs of all the trees. For classification, this is usually a majority vote; for regression, it is the average. A single decision tree is very sensitive to small changes in the training data, but averaging many trees smooths out that instability.

If we denote the predictions of $B$ trees by $T_1, T_2, \dots, T_B$, then the ensemble prediction is $\bar{T} = \frac{1}{B}\sum_{i=1}^{B} T_i$. Its variance comes out to be

$$
\mathrm{Var}(\bar{T}) = \frac{1}{B^2}\left(\sum_{i=1}^{B} \mathrm{Var}(T_i) + 2\sum_{i=1}^{B}\sum_{j=i+1}^{B} \mathrm{Cov}(T_i, T_j)\right)
$$

If all trees have the same variance $\sigma^2$ and pairwise correlation $\rho$, this becomes

$$
\mathrm{Var}(\bar{T}) = \frac{\sigma^2}{B} + \frac{B-1}{B}\rho\sigma^2
$$

This equation is the heart of the idea. The first term, $\frac{\sigma^2}{B}$, is the usual averaging effect: if the trees are roughly independent, then averaging many of them reduces the variance. The second term, $\frac{B-1}{B}\rho\sigma^2$, is the correlation penalty: as the trees become more correlated, this term gets larger and the benefit of averaging shrinks. So when the trees are all making similar predictions, the variance reduction is much weaker. This is why bagging helps: each tree is trained on a slightly different bootstrap sample, so they do not all learn the same thing.

###### Random Feature Selection

Random feature selection is what makes a random forest different from a plain bagging ensemble. When a tree is split, instead of considering every feature at each node, we randomly choose a subset of features and only search among those. This prevents the model from repeatedly using the same strongest features and forces the trees to become more diverse. If every tree used the same highly predictive feature at each split, the ensemble would still be highly correlated and would not reduce variance as effectively.

In practice, the number of features considered at each split is usually a small fraction of the total; for classification, it is often the square root of the number of features. This extra randomness decorrelates the trees and often improves generalisation. So while bagging reduces variance by averaging many trees, random feature selection reduces the correlation between those trees, which makes the ensemble more effective than a naive collection of similar trees.

Both bagging and random feature selection help ensure that correlated deviations in the data do not survive. If all the trees in the ensemble are trained on similar data and then make similar splits, then their predictions will be highly correlated, so the ensemble does not gain much by averaging them. In other words, even with many trees, if they all learn the same pattern, the variance reduction is limited. This is especially relevant when some features are strongly related to each other or when a few features dominate the split decisions.

###### Feature Importance

Once a forest is trained, it is often useful to understand which features are driving the predictions. This is one of the strengths of decision trees: they are highly interpretable, and we can examine the exact rules that influence the model. I looked at two common ways to estimate feature importance: impurity-based importance and permutation importance.

Impurity-based importance is computed directly from the tree structure. Each time a feature is used to split a node, it reduces the impurity of the child nodes compared to the parent. For a given feature, we can sum the reduction in impurity across all nodes where it was used, often weighted by the number of samples reaching that split. Features that create stronger, more useful splits are assigned larger importance scores. However, this may produce a misleading ranking when useful features are highly correlated, because only one of the correlated features may appear as the dominant split while the other carries similar information.

Permutation importance is a more performance-based check. We take a trained model, randomly shuffle one feature in the validation or test set, and measure how much the accuracy or error worsens. If the feature is important, scrambling it should cause a noticeable drop in performance. Repeating this across features gives a ranking of which variables matter most. The advantage of permutation importance is that it directly measures the effect of a feature on predictive performance, not just on the internal split logic.

###### Applications

Apparently decision trees and forests are tradtionally used for structured, tabular data like credit scoring and loan approval, fraud detection, customer churn prediction, medical risk assessment, demand forecasting, recommendation systems, and marketing decisions. 
Individual trees are useful for maximum interpretability, while random forests are useful when better predictive stability is needed. They are also often used as a strong baseline for a problem or to estimate feature importance before trying more complex models. Although other methods, such as gradient-boosted trees (XGBoost) and neural networks, may perform better in many situations, forests remain practical because while they can model nonlinear relationships, they require relatively little preprocessing, and work well on many different types of tabular data.


