### WRITE UP

##### How I learnt

I started in Aug by going through all the resources provided in the first mail. I may have forgotten the specifics by now but I got a good feel for the basic aspects of statistical learning, some of the vocabulary and metrics, and developed an interest in the field.

I chose decision trees for the task as it felt the most accessible and intuitive, even though I had spent some term learning about neural networks before that.

Before going in the research papers and the book, I watched a few yt videos to understand the basic idea, and then ended up relying on chatgpt to help me read the theory. 

For writing the code, copilot and chatgpt were a big help as i told them to guide me through the process step by step, explaining how popular code implementations work, incrementally building up the tree. After I got nice results w/ the very simple iris dataset, I decided I wanted to run it on random real world data, as that may lead to more interesting outcomes. Hence I took an IMDB dataset, and predicting the genre of a movie based on numbers like runtime and metascore should be like predicting something totally random, yet I was surprised when the tree and the forest got test accuracies around 0.6.

I definitely am not good at using Git and realized that I wasn't on the main branch for half the work I did. Anyhow I managed to make it work and show meaningful commits. I also ended up doing most of the work on a single file, but saved results from different experiments in the form of many screenshots. I also referred to the Stanford lecture for the theory.



I did not implement regression and categorical variables mainly because I wanted to get till making forests before making changes in the tree code that would take me time to understand.



I also wanted to run it on a huge dataset which theoretically should be predictable - so the 56000 x 24 lifestyle one seemed like a great option. It also made me realize the insane amount of compute power required for training. This set was so big that I simply couldn't finish training it. I had to significantly shorten it(by more than 95%) to reduce the training time for a single tree to just over an hour on my laptop. (Also made copilot add a progress check to confirm its still running)

##### How Decision Trees and Forests Work

The heart of the decision tree is segmenting the entire observation space, traditionally orthogonal, and predicting the value of a point based on the class of the segment/leaf it lies within. Since we go top down, making simple 'greater than lesser than' splits across single features at a time, the algorithm is relatively fast to train and use. The optimum split is decided at that node based on what results in the least 'information lost' or max 'information gain' depending upon which representation of information you decide to use, like GINI or Entropy. Most functions to measure information/impurity form a curve like !\[Impurity]("C:\\projects\\postman aiml\\Postman-AI-ML\\basictree\\results\\Gini Impurity vs Entropy.png")



Decisions trees are fundamentally high variance models. They are extremely sensitive to the specifics of the training data and early splits have an outsized influence on later ones. Factors like limiting the max depth, limiting the leaf size or node size for splitting and reduce variance and hence overfitting. 



The whole point of using ensembles (ie. a random forest) is to reduce the variance.

###### Bagging:

###### Random Feature Selection:



##### Learnings

