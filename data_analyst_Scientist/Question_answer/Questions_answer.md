# Data Science/Analytics Questions Interview Questions

Common Data Science questions curated from the internet.<br>
*Disclaimer* I'm not in HR.<br>
Sources
1. [edureka](https://www.edureka.co/blog/interview-questions/top-data-science-interview-questions-for-budding-data-scientists/)
2. [Towards Data Science](https://towardsdatascience.com/top-30-data-science-interview-questions-7dd9a96d3f5c)
3. [https://onlinetutorials.today](https://onlinetutorials.today/data-science/data-science-interview-questions-and-answers/)
4. [dezyre.com](https://www.dezyre.com/article/data-analyst-interview-questions-to-prepare-for-in-2018/324)
<br>

<details><summary><b>What is Data Science? Also, list the differences between supervised and unsupervised learning.. </b></summary>

 > Data science is a multi-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.

 > supervised learning
   * Input data is labeled.
   * Uses training dataset.
   * Used for prediction.
   * Enables classification and regression.
 > Unsupervised Learning
   * Input data is unlabeled.
   * Uses the input data set.
   * Used for analysis.
   * Enables Classification, Density Estimation, & Dimension Reduction

 > Supervised learning: Supervised learning is the learning of the model where with input variable ( say, x) and an output variable (say, Y) and an algorithm to map the input to the output.
 That is, Y = f(X) 

 > Unsupervised learning is where only the input data (say, X) is present and no corresponding output variable is there.


</details>


<details><summary><b> What are the important skills to have in Python with regard to data analysis? </b></summary>

 >  * Good understanding of the built-in data types especially lists, dictionaries, tuples, and sets.
    * Mastery of N-dimensional NumPy Arrays.
    * Mastery of Pandas dataframes.
    * Ability to perform element-wise vector and matrix operations on NumPy arrays.
    * Knowing that you should use the Anaconda distribution and the conda package manager.
    * Familiarity with Scikit-learn. **Scikit-Learn Cheat Sheet**
    * Ability to write efficient list comprehensions instead of traditional for loops.
    * Ability to write small, clean functions (important for any developer), preferably pure functions that don’t alter objects.
    * Knowing how to profile the performance of a Python script and how to optimize bottlenecks.



</details>

<details><summary><b>  What is Selection Bias? </b></summary>

 > Selection bias is a kind of error that occurs when the researcher decides who is going to be studied. It is usually associated with research where the selection of participants isn’t random. It is sometimes referred to as the selection effect. It is the distortion of statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate.

 > The types of selection bias include:

    Sampling bias: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.

    Time interval: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.

    Data: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.

    Attrition: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.

</details>


<details><summary><b> What is the difference between “long” and “wide” format data? </b></summary>

 > In the wide format, a subject’s repeated responses will be in a single row, and each response is in a separate column. In the long format, each row is a one-time point per subject. You can recognize data in wide format by the fact that columns generally represent groups.

</details>


<details><summary><b> What do you understand by the term Normal Distribution? </b></summary>

 > Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled up.

    However, there are chances that data is distributed around a central value without any bias to the left or right and reaches normal distribution in the form of a bell-shaped curve.
    The random variables are distributed in the form of a symmetrical bell-shaped curve.

    Properties of Nornal Distribution:

    Unimodal -one mode
    Symmetrical -left and right halves are mirror images
    Bell-shaped -maximum height (mode) at the mean
    Mean, Mode, and Median are all located in the center
    Asymptotic

</details>

<details><summary><b> What is the goal of A/B Testing? </b></summary>

 > It is a statistical hypothesis testing for a randomized experiment with two variables A and B.

    The goal of A/B Testing is to identify any changes to the web page to maximize or increase the outcome of an interest. A/B testing is a fantastic method for figuring out the best online promotional and marketing strategies for your business. It can be used to test everything from website copy to sales emails to search ads

</details>


<details><summary><b> What do you understand by statistical power of sensitivity and how do you calculate it? </b></summary>

 > Sensitivity is commonly used to validate the accuracy of a classifier (Logistic, SVM, Random Forest etc.).

    Sensitivity is nothing but “Predicted True events/ Total events”. True events here are the events which were true and model also predicted them as true.

    Calculation of seasonality 

    Seasonality = ( True Positives ) / ( Positives in Actual Dependent Variable )

</details>


<details><summary><b> What are the differences between overfitting and underfitting? </b></summary>

 > Overfitting occurs when a statistical model or machine learning algorithm captures the noise of the data.  Intuitively, overfitting occurs when the model or the algorithm fits the data too well.  Specifically, overfitting occurs if the model or algorithm shows low bias but high variance.  Overfitting is often a result of an excessively complicated model, and it can be prevented by fitting multiple models and using validation or cross-validation to compare their predictive accuracies on test data.

 > Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data.  Intuitively, underfitting occurs when the model or the algorithm does not fit the data well enough.  Specifically, underfitting occurs if the model or algorithm shows low variance but high bias.  Underfitting is often a result of an excessively simple model.

</details>

<details><summary><b>  Python or R – Which one would you prefer for text analytics? </b></summary>

 > We will prefer Python because of the following reasons:

    Python would be the best option because it has Pandas library that provides easy to use data structures and high-performance data analysis tools.
    R is more suitable for machine learning than just text analysis.
    Python performs faster for all types of text analytics.

</details>

<details><summary><b> How does data cleaning plays a vital role in analysis? </b></summary>

 > Data cleaning can help in analysis because:

    Cleaning data from multiple sources helps to transform it into a format that data analysts or data scientists can work with.
    Data Cleaning helps to increase the accuracy of the model in machine learning.
    It is a cumbersome process because as the number of data sources increases, the time taken to clean the data increases exponentially due to the number of sources and the volume of data generated by these sources.
    It might take up to 80% of the time for just cleaning data making it a critical part of analysis task.

</details>

<details><summary><b> Differentiate between univariate, bivariate and multivariate analysis. </b></summary>

 > Univariate analyses are descriptive statistical analysis techniques which can be differentiated based on the number of variables involved at a given point of time. For example, the pie charts of sales based on territory involve only one variable and can the analysis can be referred to as univariate analysis.

 > The bivariate analysis attempts to understand the difference between two variables at a time as in a scatterplot. For example, analyzing the volume of sale and spending can be considered as an example of bivariate analysis.

 > Multivariate analysis deals with the study of more than two variables to understand the effect of variables on the responses.

</details>


<details><summary><b> What is Cluster Sampling? </b></summary>

 > Cluster sampling refers to a type of sampling method . With cluster sampling, the researcher divides the population into separate groups, called clusters. Then, a simple random sample of clusters is selected from the population. The researcher conducts his analysis on data from the sampled clusters.

</details>


<details><summary><b> What is Systematic Sampling? </b></summary>

 > Systematic sampling is a statistical technique where elements are selected from an ordered sampling frame. In systematic sampling, the list is progressed in a circular manner so once you reach the end of the list, it is progressed from the top again. The best example of systematic sampling is equal probability method.

</details>


<details><summary><b> What are Eigenvectors and Eigenvalues? </b></summary>

 > Eigenvectors are used for understanding linear transformations. In data analysis, we usually calculate the eigenvectors for a correlation or covariance matrix. Eigenvectors are the directions along which a particular linear transformation acts by flipping, compressing or stretching.

 > Eigenvalue can be referred to as the strength of the transformation in the direction of eigenvector or the factor by which the compression occurs.

</details>

<details><summary><b> Can you cite some examples where a false positive is important than a false negative? </b></summary>

 >      False Positives are the cases where you wrongly classified a non-event as an event a.k.a Type I error.
    False Negatives are the cases where you wrongly classify events as non-events, a.k.a Type II error.

 > A false positive is where you receive a positive result for a test, when you should have received a negative results
 > False Negatives, you get a negative test result, but you should have got a positive test result.

</details>


<details><summary><b> Can you cite some examples where both false positive and false negatives are equally important? </b></summary>

 > false positives:

    A pregnancy test is positive, when in fact you aren’t pregnant.
    A cancer screening test comes back positive, but you don’t have the disease.
    A prenatal test comes back positive for Down’s Syndrome, when your fetus does not have the disorder(1).
    Virus software on your computer incorrectly identifies a harmless program as a malicious one.

 > False Negative
    Quality control in manufacturing; a false negative in this area means that a defective item passes through the cracks.
    In software testing, a false negative would mean that a test designed to catch something (i.e. a virus) has failed.
    In the Justice System, a false negative occurs when a guilty suspect is found “Not Guilty” and allowed to walk free.



</details>


<details><summary><b>Can you explain the difference between a Validation Set and a Test Set? </b></summary>

 > A Validation set can be considered as a part of the training set as it is used for parameter selection and to avoid overfitting of the model being built. On the other hand, a Test Set is used for testing or evaluating the performance of a trained machine learning model.

</details>


<details><summary><b> Explain cross-validation. </b></summary>

 > Cross-validation is a model validation technique for evaluating how the outcomes of statistical analysis will generalize to an Independent dataset. Mainly used in backgrounds where the objective is forecast and one wants to estimate how accurately a model will accomplish in practice. The goal of cross-validation is to term a data set to test the model in the training phase (i.e. validation data set) in order to limit problems like overfitting and get an insight on how the model will generalize to an independent data set.

</details>


<details><summary><b> What is Machine Learning? </b></summary>

 > Machine Learning explores the study and construction of algorithms that can learn from and make predictions on data. Closely related to computational statistics. Used to devise complex models and algorithms that lend themselves to a prediction which in commercial use is known as predictive analytics.

</details>

<details><summary><b> What is logistic regression? State an example when you have used logistic regression recently. </b></summary>

 > Logistic Regression often referred as logit model is a technique to predict the binary outcome from a linear combination of predictor variables. 

 For example, if you want to predict whether a particular political leader will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The predictor variables here would be the amount of money spent for election campaigning of a particular candidate, the amount of time spent in campaigning, etc.

</details>


<details><summary><b> What are Recommender Systems? </b></summary>

 > Recommender Systems are a subclass of information filtering systems that are meant to predict the preferences or ratings that a user would give to a product. Recommender systems are widely used in movies, news, research articles, products, social tags, music, etc.

 Examples include movie recommenders in IMDB, Netflix & BookMyShow, product recommenders in e-commerce sites like Amazon, eBay & Flipkart, YouTube video recommendations and game recommendations in Xbox.


</details>


<details><summary><b> What is Linear Regression? </b></summary>

 > Linear regression is a statistical technique where the score of a variable Y is predicted from the score of a second variable X. X is referred to as the predictor variable and Y as the criterion variable.

</details>


<details><summary><b> What is Collaborative filtering? </b></summary>

 > The process of filtering used by most of the recommender systems to find patterns or information by collaborating viewpoints, various data sources and multiple agents.

</details>



<details><summary><b>How can outlier values be treated? </b></summary>

 > Outlier values can be identified by using univariate or any other graphical analysis method. If the number of outlier values is few then they can be assessed individually but for a large number of outliers, the values can be substituted with either the 99th or the 1st percentile values.

 All extreme values are not outlier values. The most common ways to treat outlier values

    To change the value and bring in within a range.
    To just remove the value.

</details>


<details><summary><b> What are the various steps involved in an analytics project? </b></summary>

 > The following are the various steps involved in an analytics project:

    * Understand the Business problem
    * Explore the data and become familiar with it.
    * Prepare the data for modeling by detecting outliers, treating missing values, transforming variables, etc.
    * After data preparation, start running the model, analyze the result and tweak the approach. This is an iterative step until the best possible outcome is achieved.
    * Validate the model using a new data set.
    * Start implementing the model and track the result to analyze the performance of the model over the period of time.

</details>


<details><summary><b>  During analysis, how do you treat missing values? </b></summary>

 > The extent of the missing values is identified after identifying the variables with missing values. If any patterns are identified the analyst has to concentrate on them as it could lead to interesting and meaningful business insights.

 > If there are no patterns identified, then the missing values can be substituted with mean or median values (imputation) or they can simply be ignored. Assigning a default value which can be mean, minimum or maximum value. Getting into the data is important.

 > If it is a categorical variable, the default value is assigned. The missing value is assigned a default value. If you have a distribution of data coming, for normal distribution give the mean value.

 > If 80% of the values for a variable are missing then you can answer that you would be dropping the variable instead of treating the missing values.

</details>


<details><summary><b> What do you mean by Deep Learning and Why has it become popular now? </b></summary>

 > Deep Learning is nothing but a paradigm of machine learning which has shown incredible promise in recent years. This is because of the fact that Deep Learning shows a great analogy with the functioning of the human brain.

 Now although Deep Learning has been around for many years, the major breakthroughs from these techniques came just in recent years. This is because of two main reasons:

    The increase in the amount of data generated through various sources
    The growth in hardware resources required to run these models

 GPUs are multiple times faster and they help us build bigger and deeper deep learning models in comparatively less time than we required previously

</details>

<details><summary><b> What are Artificial Neural Networks? </b></summary>

 > Artificial Neural networks are a specific set of algorithms that have revolutionized machine learning. They are inspired by biological neural networks. Neural Networks can adapt to changing input so the network generates the best possible result without needing to redesign the output criteria.

</details>


<details><summary><b> Describe the structure of Artificial Neural Networks? </b></summary>

 > Artificial Neural Networks works on the same principle as a biological Neural Network. It consists of inputs which get processed with weighted sums and Bias, with the help of Activation Functions.

</details>

<details><summary><b> Explain Gradient Descent. </b></summary>

 > To Understand Gradient Descent, Let’s understand what is a Gradient first.

 > A gradient measures how much the output of a function changes if you change the inputs a little bit. It simply measures the change in all weights with regard to the change in error. You can also think of a gradient as the slope of a function.

 > Gradient Descent can be thought of climbing down to the bottom of a valley, instead of climbing up a hill.  This is because it is a minimization algorithm that minimizes a given function (Activation Function).



</details>


<details><summary><b>What is Back Propagation and Explain it’s Working. </b></summary>

 > Backpropagation is a training algorithm used for multilayer neural network. In this method, we move the error from an end of the network to all weights inside the network and thus allowing efficient computation of the gradient.

 It has the following steps:

    Forward Propagation of Training Data
    Derivatives are computed using output and target
    Back Propagate for computing derivative of error wrt output activation
    Using previously calculated derivatives for output
    Update the Weights

</details>

<details><summary><b> What are the variants of Back Propagation? </b></summary>

 > *  Stochastic Gradient Descent: We use only single training example for calculation of gradient and update parameters.
    *  Batch Gradient Descent: We calculate the gradient for the whole dataset and perform the update at each iteration.
    *  Mini-batch Gradient Descent: It’s one of the most popular optimization algorithms. It’s a variant of Stochastic Gradient Descent and here instead of single training example, mini-batch of samples is used.

</details>

<details><summary><b>What are the different Deep Learning Frameworks? </b></summary>

 >  Pytorch
    TensorFlow
    Microsoft Cognitive Toolkit
    Keras
    Caffe
    Chainer

</details>

<details><summary><b> What is the role of Activation Function? </b></summary>

 > The Activation function is used to introduce non-linearity into the neural network helping it to learn more complex function. Without which the neural network would be only able to learn linear function which is a linear combination of its input data. An activation function is a function in an artificial neuron that delivers an output based on inputs

</details>


<details><summary><b>  What is an Auto-Encoder?  </b></summary>

 > Autoencoders are simple learning networks that aim to transform inputs into outputs with the minimum possible error. This means that we want the output to be as close to input as possible. We add a couple of layers between the input and the output, and the sizes of these layers are smaller than the input layer. The autoencoder receives unlabeled input which is then encoded to reconstruct the input.

</details>

<details><summary><b> What is a Boltzmann Machine?  </b></summary>

 > Boltzmann machines have a simple learning algorithm that allows them to discover interesting features that represent complex regularities in the training data. The Boltzmann machine is basically used to optimize the weights and the quantity for the given problem. The learning algorithm is very slow in networks with many layers of feature detectors. “Restricted Boltzmann Machines” algorithm has a single layer of feature detectors which makes it faster than the rest.

</details>


<details><summary><b> What is bias, variance trade off ? </b></summary>

 > “Bias is error introduced in your model due to over simplification of machine learning algorithm.” It can lead to under fitting. When you train your model at that time model makes simplified assumptions to make the target function easier to understand.

    Low bias machine learning algorithms — Decision Trees, k-NN and SVM High bias machine learning algorithms — Linear Regression, Logistic Regression

 >  “Variance is error introduced in your model due to complex machine learning algorithm, your model learns noise also from the training data set and performs bad on test data set.” It can lead high sensitivity and over fitting.

 > Normally, as you increase the complexity of your model, you will see a reduction in error due to lower bias in the model. However, this only happens till a particular point. As you continue to make your model more complex, you end up over-fitting your model and hence your model will start suffering from high variance.

 > Bias, Variance trade off: The goal of any supervised machine learning algorithm is to have low bias and low variance to achieve good prediction performance.

    The k-nearest neighbours algorithm has low bias and high variance, but the trade-off can be changed by increasing the value of k which increases the number of neighbours that contribute to the prediction and in turn increases the bias of the model.
    The support vector machine algorithm has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.

 There is no escaping the relationship between bias and variance in machine learning. Increasing the bias will decrease the variance. Increasing the variance will decrease the bias.

</details>



<details><summary><b>What is exploding gradients ? </b></summary>

 > Gradient is the direction and magnitude calculated during training of a neural network that is used to update the network weights in the right direction and by the right amount.

 > “Exploding gradients are a problem where large error gradients accumulate and result in very large updates to neural network model weights during training.” At an extreme, the values of weights can become so large as to overflow and result in NaN values.

 > This has the effect of your model being unstable and unable to learn from your training data. Now let’s understand what is the gradient.

</details>


<details><summary><b>  What is a confusion matrix ? </b></summary>

 > The confusion matrix is a 2X2 table that contains 4 outputs provided by the binary classifier. Various measures, such as error-rate, accuracy, specificity, sensitivity, precision and recall are derived from it. Confusion Matrix

</details>


<details><summary><b>Explain how a ROC curve works ? </b></summary>

 > The ROC curve is a graphical representation of the contrast between true positive rates and false positive rates at various thresholds. It is often used as a proxy for the trade-off between the sensitivity(true positive rate) and false positive rate.

</details>

<details><summary><b> What is selection Bias ? </b></summary>

 > Selection bias occurs when sample obtained is not representative of the population intended to be analysed.

</details>

<details><summary><b>Explain SVM machine learning algorithm in detail. </b></summary>

 > SVM stands for support vector machine, it is a supervised machine learning algorithm which can be used for both Regression and Classification. If you have n features in your training data set, SVM tries to plot it in n-dimensional space with the value of each feature being the value of a particular coordinate. SVM uses hyper planes to separate out different classes based on the provided kernel function.

</details>


<details><summary><b>What are the different kernels functions in SVM ?  </b></summary>

 > There are four types of kernels in SVM.

    Linear Kernel
    Polynomial kernel
    Radial basis kernel
    Sigmoid kernel

</details>



<details><summary><b>What is Entropy and Information gain in Decision tree algorithm ? </b></summary>

 > Entropy
    A decision tree is built top-down from a root node and involve partitioning of data into homogenious subsets. ID3 uses enteropy to check the homogeneity of a sample. If the sample is completely homogenious then entropy is zero and if the sample is an equally divided it has entropy of one.

 > The Information Gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding attributes that returns the highest information gain.

</details>


<details><summary><b>Explain Decision Tree algorithm in detail. </b></summary>

 > Decision tree is a supervised machine learning algorithm mainly used for the Regression and Classification.It breaks down a data set into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. Decision tree can handle both categorical and numerical data.

</details>

<details><summary><b>What is pruning in Decision Tree ? </b></summary>

 > When we remove sub-nodes of a decision node, this process is called pruning or opposite process of splitting.

</details>


<details><summary><b> What is Ensemble Learning ? </b></summary>

 > Ensemble is the art of combining diverse set of learners(Individual models) together to improvise on the stability and predictive power of the model. Ensemble learning has many types but two more popular ensemble learning techniques are mentioned below.

     Bagging

    Bagging tries to implement similar learners on small sample populations and then takes a mean of all the predictions. In generalised bagging, you can use different learners on different population. As you expect this helps us to reduce the variance error.

    Boosting

    Boosting is an iterative technique which adjust the weight of an observation based on the last classification. If an observation was classified incorrectly, it tries to increase the weight of this observation and vice versa. Boosting in general decreases the bias error and builds strong predictive models. However, they may over fit on the training data.

</details>


<details><summary><b> What is Random Forest? How does it work ? </b></summary>

 > Random forest is a versatile machine learning method capable of performing both regression and classification tasks. It is also used for dimentionality reduction, treats missing values, outlier values. It is a type of ensemble learning method, where a group of weak models combine to form a powerful model.

> In Random Forest, we grow multiple trees as opposed to a single tree. To classify a new object based on attributes, each tree gives a classification. The forest chooses the classification having the most votes(Over all the trees in the forest) and in case of regression, it takes the average of outputs by different trees.

</details>


<details><summary><b> What cross-validation technique would you use on a time series data set.</b></summary>

 > Instead of using k-fold cross-validation, you should be aware to the fact that a time series is not randomly distributed data — It is inherently ordered by chronological order.

> In case of time series data, you should use techniques like forward chaining — Where you will be model on past data then look at forward-facing data.

    fold 1: training[1], test[2]

    fold 1: training[1 2], test[3]

    fold 1: training[1 2 3], test[4]

    fold 1: training[1 2 3 4], test[5]

</details>


<details><summary><b> What is logistic regression? Or State an example when you have used logistic regression recently. </b></summary>

 > Logistic Regression often referred as logit model is a technique to predict the binary outcome from a linear combination of predictor variables. For example, if you want to predict whether a particular political leader will win the election or not. In this case, the outcome of prediction is binary i.e. 0 or 1 (Win/Lose). The predictor variables here would be the amount of money spent for election campaigning of a particular candidate, the amount of time spent in campaigning, etc.

</details>

<details><summary><b> What is a Box Cox Transformation? </b></summary>

 > Dependent variable for a regression analysis might not satisfy one or more assumptions of an ordinary least squares regression. The residuals could either curve as the prediction increases or follow skewed distribution. In such scenarios, it is necessary to transform the response variable so that the data meets the required assumptions. A Box cox transformation is a statistical technique to transform non-normal dependent variables into a normal shape. If the given data is not normal then most of the statistical techniques assume normality. Applying a box cox transformation means that you can run a broader number of tests.

</details>


<details><summary><b> What is deep learning? </b></summary>

 > Deep learning is sub field of machine learning inspired by structure and function of brain called artificial neural network. We have a lot numbers of algorithms under machine learning like Linear regression, SVM, Neural network etc and deep learning is just an extension of Neural networks. In neural nets we consider small number of hidden layers but when it comes to deep learning algorithms we consider a huge number of hidden layers to better understand the input output relationship.

</details>


<details><summary><b> What are Recurrent Neural Networks(RNNs) ? </b></summary>

 > Recurrent nets are type of artificial neural networks designed to recognise pattern from the sequence of data such as Time series, stock market and government agencies etc. To understand recurrent nets, first you have to understand the basics of feed forward nets. Both these networks RNN and feed forward named after the way they channel information through a series of mathematical orations performed at the nodes of the network. One feeds information through straight(never touching same node twice), while the other cycles it through loop, and the latter are called recurrent.
> Recurrent networks on the other hand, take as their input not just the current input example they see, but also the what they have perceived previously in time. The BTSXPE at the bottom of the drawing represents the input example in the current moment, and CONTEXT UNIT represents the output of the previous moment. The decision a recurrent neural network reached at time t-1 affects the decision that it will reach one moment later at time t. So recurrent networks have two sources of input, the present and the recent past, which combine to determine how they respond to new data, much as we do in life.

> The error they generate will return via back propagation and be used to adjust their weights until error can’t go any lower. Remember, the purpose of recurrent nets is to accurately classify sequential input. We rely on the back propagation of error and gradient descent to do so.

Back propagation in feed forward networks moves backward from the final error through the outputs, weights and inputs of each hidden layer, assigning those weights responsibility for a portion of the error by calculating their partial derivatives — ∂E/∂w, or the relationship between their rates of change. Those derivatives are then used by our learning rule, gradient descent, to adjust the weights up or down, whichever direction decreases error.

Recurrent networks rely on an extension of back propagation called back propagation through time, or BPTT. Time, in this case, is simply expressed by a well-defined, ordered series of calculations linking one time step to the next, which is all back propagation needs to work.

</details>



<details><summary><b> What is the difference between machine learning and deep learning? </b></summary>

 > Machine learning:

Machine learning is a field of computer science that gives computers the ability to learn without being explicitly programmed. Machine learning can be categorised in following three categories.

    Supervised machine learning,
    Unsupervised machine learning,
    Reinforcement learning

Deep learning:

Deep Learning is a sub field of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

</details>

<details><summary><b> What is reinforcement learning ? </b></summary>

 > Reinforcement Learning is learning what to do and how to map situations to actions. The end result is to maximise the numerical reward signal. The learner is not told which action to take, but instead must discover which action will yield the maximum reward.Reinforcement learning is inspired by the learning of human beings, it is based on the reward/panelity mechanism.

</details>


<details><summary><b> What is selection bias ? </b></summary>

 > Selection bias is the bias introduced by the selection of individuals, groups or data for analysis in such a way that proper randomisation is not achieved, thereby ensuring that the sample obtained is not representative of the population intended to be analysed. It is sometimes referred to as the selection effect. The phrase “selection bias” most often refers to the distortion of a statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may not be accurate.

</details>


<details><summary><b> Explain what regularisation is and why it is useful. </b></summary>

 > Regularisation is the process of adding tunning parameter to a model to induce smoothness in order to prevent overfitting. This is most often done by adding a constant multiple to an existing weight vector. This constant is often the L1(Lasso) or L2(ridge). The model predictions should then minimize the loss function calculated on the regularized training set.

</details>

<details><summary><b> What is TF/IDF vectorization ? </b></summary>

 > tf–idf is short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining. The tf-idf value increases proportionally to the number of times a word appears in the document, but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

</details>


<details><summary><b> What is the difference between Regression and classification ML techniques.</b></summary>

 > Both Regression and classification machine learning techniques come under Supervised machine learning algorithms. In Supervised machine learning algorithm, we have to train the model using labelled data set, While training we have to explicitly provide the correct labels and algorithm tries to learn the pattern from input to output. If our labels are discrete values then it will a classification problem, e.g A,B etc. but if our labels are continuous values then it will be a regression problem, e.g 1.23, 1.333 etc.

</details>





<details><summary><b> If you are having 4GB RAM in your machine and you want to train your model on 10GB data set. How would you go about this problem. Have you ever faced this kind of problem in your machine learning/data science experience so far ? </b></summary>

 > First of all you have to ask which ML model you want to train.

For Neural networks: Batch size with Numpy array will work.

Steps:

    Load the whole data in Numpy array. Numpy array has property to create mapping of complete data set, it doesn’t load complete data set in memory.
    You can pass index to Numpy array to get required data.
    Use this data to pass to Neural network.
    Have small batch size.

For SVM: Partial fit will work

Steps:

    Divide one big data set in small size data sets.
    Use partial fit method of SVM, it requires subset of complete data set.
    Repeat step 2 for other subsets.

</details>



<details><summary><b> What is p-value? </b></summary>

 > When you perform a hypothesis test in statistics, a p-value can help you determine the strength of your results. p-value is a number between 0 and 1. Based on the value it will denote the strength of the results. The claim which is on trial is called Null Hypothesis.

</details>



<details><summary><b>What is ‘Naive’ in a Naive Bayes ? </b></summary>

 > The Naive Bayes Algorithm is based on the Bayes Theorem. Bayes’ theorem describes the probability of an event, based on prior knowledge of conditions that might be related to the event.

</details>


#### Statistics interview questions:

<details><summary><b> What is Statistics ?</b></summary>

 > It is a branch of mathematics pertaining to the collection, analysis, interpretation, and presentation of masses of numerical data.

</details>



<details><summary><b> How many Types of statistics are there ? </b></summary>

 >  Descriptive Statistics
 >  Inferential Statistics


</details>



<details><summary><b> What is Descriptive statistics ? </b></summary>

 > It is help to organize data and focus on the main characteristic of the data and it’s also provides a summary of he data numerically and graphically. (mean, mode, standard deviation, correlation)

</details>



<details><summary><b> What is inferential statistics ? </b></summary>

 > It generates the larger data and applies probability theory to draw a conclusion

</details>


<details><summary><b> What is mean value in statistics ? </b></summary>

 > Mean is the average value of the data set.

</details>



<details><summary><b> What is Mode value in statistics ?  </b></summary>

 > The Most repeated value in the data set

</details>



<details><summary><b> What is median value in statistics ?  </b></summary>

 > The middle value from data set

</details>



<details><summary><b> What is Variance in statistics ?  </b></summary>

 > Variance measures how far each number in the set is from the mean.

</details>


<details><summary><b> What is standard Deviation in statistics ? </b></summary>

 > It is a square root of variance

</details>


<details><summary><b> How many types of variables are there in statistics ? </b></summary>

 > 
    Categorical variable
    Confounding variable
    Continuous variable
    Control variable
    Dependent variable
    Discrete variable
    Independent variable
    Nominal variable
    Ordinal variable
    Qualitative variable
    Quantitative variable
    Random variables
    Ratio variables
    ranked variables


</details>


<details><summary><b> How many types of distributions are there ? </b></summary>

 > 
    Bernoulli Distribution
    Uniform Distribution
    Binomial Distribution
    Normal Distribution
    Poisson Distribution
    Exponential Distribution


</details>


<details><summary><b> What is normal distribution ? </b></summary>

 > A) It’s like a bell curve distribution. Mean, Mode and Medium are equal in this distribution. Most of the distributions in statistics are normal distribution.



</details>

<details><summary><b> What is standard normal distribution ?  </b></summary>

 > If mean is 0 and standard deviation is 1 then we call that distribution as standard normal distribution.

</details>



<details><summary><b> What is Binominal Distribution ? </b></summary>

 > A distribution where only two outcomes are possible, such as success or failure and where the probability of success and failure is same for all the trials then it is called a Binomial Distribution

</details>



<details><summary><b> What is Bernoulli distribution ? </b></summary>

 > A Bernoulli distribution has only two possible outcomes, namely 1 (success) and 0 (failure), and a single trial.

</details>


<details><summary><b> What is Poisson distribution ? </b></summary>

 > A distribution is called Poisson distribution when the following assumptions are true:

1. Any successful event should not influence the outcome of another successful event.
2. The probability of success over a short interval must equal the probability of success over a longer interval.
3. The probability of success in an interval approaches zero as the interval becomes smaller.

</details>


<details><summary><b> What is central limit theorem ? </b></summary>

> a) Mean of sample means is closely to the mean of the population

> b) Standard deviation of the sample distribution can be found out from the population standard deviation divided by square root of sample size N and it is also known as standard error of means.

> c) if the population is not normal distribution, but the sample size is greater than 30 the sampling distribution of sample means approximates a normal distribution

</details>


<details><summary><b>What is P Value, How it’s useful ? </b></summary>

 > The p-value is the level of marginal significance within a statistical hypothesis test representing the probability of the occurrence of a given event.

    If The p-value is  less than 0.05 (p<=0.05), It indicates strong evidence against the null hypothesis, you can reject the Null Hypothesis
    If the P-value is higher than 0.05 (p>0.05), It indicates weak evidence against the null hypothesis, you can fail to reject the null Hypothesis


</details>


<details><summary><b> What is Z value or Z score (Standard Score)  , How it’s useful ? </b></summary>

 > Z score indicates how many standard deviations on element is from the mean. It is also called standard score.

Z score Formula

z = (X – μ) / σ

    It is useful in Statistical testing.
    Z-value is range between -3 to 3.
    Its useful to find the outliers in large data


</details>


<details><summary><b>What is T-Score, What is the use of it ? </b></summary>

 > 
    It  is a ratio between the difference between two groups and the difference within the groups. The larger t score, the more difference there is between groups. The smaller t-score means the more similarity between groups.
    We can use t-score when the sample size is less than 30, It is used in statistical testing


</details>

<details><summary><b>What is IQR ( Interquartile Range ) and Usage ? </b></summary>

 > 
    It is difference between 75th and 25th percentiles, or between upper and lower quartiles,
    It is also called Misspread data or Middle 50%.
    Mainly to find outliers in data, if the observations that fall below Q1 − 1.5 IQR or above Q3 + 1.5 IQR those are considered as outliers.


</details>


<details><summary><b> What is Hypothesis Testing ?</b></summary>

 > Hypothesis testing is a statistical method that is used in making statistical decisions using experimental data. Hypothesis Testing is basically an assumption that we make about the population parameter.

</details>


<details><summary><b>How many Types of Hypothesis Testing are there ? </b></summary>

 > Null Hypothesis, Alternative Hypothesis

</details>



<details><summary><b> What is Type 1 Error ? </b></summary>

 > FP – False Positive ( In statistics it is the rejection of a true null hypothesis)

</details>


<details><summary><b> What is Type 2 Error ?  </b></summary>

 > FN – False Negative  ( In statistics it is failing to reject a false null hypothesis)

</details>



<details><summary><b> What is population ? </b></summary>

 > It is a discrete group of people, animals or things that can be identified by at least one common characteristic for the purposes of data collection and analysis

</details>



<details><summary><b> What is sampling ?</b></summary>

 > Sampling is a process used in statistical analysis in which a predetermined number of observations are taken from a larger population.

</details>



<details><summary><b> Types of sampling techniques ? </b></summary>

 There are two major types of sampling
1. PROBABILITY SAMPLING

    Simple Random Sampling
    Stratified Random Sampling
    Systematic Sampling
    Cluster Sampling
    Multi-stage Sampling

2. NON-PROBABILITY SAMPLING

    Purposive Sampling
    Convenience Sampling
    Snow-ball Sampling
    Quota Sampling


</details>



<details><summary><b> What is Sample Bias ?  </b></summary>

 > It is a type of bias caused by choosing non-random data for statistical analysis

</details>



<details><summary><b> What is Selection Bias ?  </b></summary>

 > Selection bias is usually introduced as an error with the sampling and having a selection for analysis that is not properly randomized

</details>



<details><summary><b> What is Univariate, Bivariate, Multi Variate Analysis ? </b></summary>

> Univarite means single variable – Analysis on single variable data

> Bivariate means two variables – you can do analysis on multiple variables

> Mutli Variate means multiple variables – Analysis on multiple variables

</details>


##### Data Science and Machine learning  Interview Questions: 

<details><summary><b> What is data science ? </b></summary>

 > Data science is the study of where information comes from, what it represents and how it can be turned into a valuable resource in the creation of business and IT strategies. Mining large amounts of structured and unstructured data to identify patterns can help an organization rein in costs, increase efficiencies, recognize new market opportunities and increase the organization’s competitive advantage.

</details>



<details><summary><b> What is Machine learning ? </b></summary>

 > Machine learning is the scientific study of algorithms and statistical models that computer systems use to progressively improve their performance on a specific task

</details>



<details><summary><b>What is Deep learning ? </b></summary>

 > Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

</details>


<details><summary><b> What is Supervised learning ? </b></summary>

 > The data is labeled.  And the algorithms learn from data to predict the output. Then we call it as supervised learning.

</details>




<details><summary><b> What is Unsupervised learning ? </b></summary>

 > Unsupervised learning is a branch of machine learning that learns from test data that has not been labeled, classified or categorized.

</details>



<details><summary><b> What is Reinforcement learning ? </b></summary>

 > Reinforcement learning is an area of machine learning concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward

</details>




<details><summary><b> What is Transfer learning ? </b></summary>

 > Transfer learning make use of the knowledge gained while solving one problem and applying it to a different but related problem.

</details>


<details><summary><b>What is Regression ? </b></summary>

 > In Statistics, a measure of the relation between the mean value of one variable (e.g. output) and corresponding values of other variables

</details>



<details><summary><b> What is Classification ? </b></summary>

 > In machine learning and statistics, classification is the problem of identifying to which of a set of categories a new observation belongs, on the basis of a training set of data containing observations whose category membership is known

</details>



<details><summary><b> What is Clustering ?</b></summary>

 > Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups

</details>



<details><summary><b> What is Bias ?  </b></summary>

 > Bias is the difference between the average prediction of our model and the correct value which we are trying to predict.

</details>


<details><summary><b> What is Variance. ? </b></summary>

 > Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.

</details>


<details><summary><b> What is EDA ? </b></summary>

 > Exploratory data analysis : In statistics, exploratory data analysis is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.

</details>


<details><summary><b> What is Overfitting, Underfitting and Trade-off ? </b></summary>

 > Overfitting – The model works fine on training data not performing well on test data.

> Underfitting- The model not able to understand patters in data

> Trade-off – We need to balance bias and variance

</details>


<details><summary><b> What are steps in Building a Machine learning Model ? </b></summary>

 > 
    Problem Statement
    Gathering Data
    Data Preparation
    EDA
    Model Training
    Validation
    Performance Tuning
    Model Deployment


</details>



<details><summary><b> What is Data Pre-processing ?</b></summary>

 > Data preprocessing is an important step in the data mining process. The phrase “garbage in, garbage out” is particularly applicable to data mining and machine learning projects. Data-gathering methods are often loosely controlled, resulting in out-of-range values, impossible data combinations, missing values, etc

</details>



<details><summary><b> What is Data Cleaning ? </b></summary>

 > Data cleansing or data cleaning is the process of detecting and correcting corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data.

</details>



<details><summary><b> What is Data Preparation ?</b></summary>

 > Data preparation is the process of cleaning and transforming raw data prior to processing and analysis. It is an important step prior to processing and often involves reformatting data, making corrections to data and the combining of data sets to enrich data.

</details>





<details><summary><b> What is Data munging ?</b></summary>

 > Data Munging is basically the hip term for cleaning up a messy data set.

</details>



<details><summary><b> What is Standardization and normalization ? </b></summary>

 > Converting variables from different ranges to same scale

</details>


<details><summary><b> How to deal with Missing Values In Data ?</b></summary>

 > It’s depends on type of data, you can fill with mean or median values, if the missing data is very less you can remove.

</details>



<details><summary><b> How to find outliers in data ? </b></summary>

 > You can find outliers in data by using box plot graphs, If the data is large, we can z values range from -3 to 3, We can also find using IQR -1.5 to 1.5.

</details>



<details><summary><b> How many types of Regression algorithms are there ? </b></summary>

 > 
    Linear Regression
    Logistic Regression
    Polynomial Regression
    Stepwise Regression
    Ridge Regression
    Lasso Regression
    ElasticNet Regression


</details>



<details><summary><b> What is Linear Regression, How it works, When to Use ? </b></summary>

 > Linear Regression can be considered a Machine Learning algorithm that allows us to map numeric inputs to numeric outputs, fitting a line into the data points.

> In other words, Linear Regression is a way of modelling the relationship between one or more variables. From the Machine Learning perspective, this is done to ensure generalization — giving the model the ability to predict outputs for inputs it has never seen before.

> Linear regression has many practical uses. Most applications fall into one of the following two broad categories:

    If the goal is prediction, or forecasting, or error reduction,[clarification needed] linear regression can be used to fit a predictive model to an observed data set of values of the response and explanatory variables. After developing such a model, if additional values of the explanatory variables are collected without an accompanying response value, the fitted model can be used to make a prediction of the response.
    If the goal is to explain variation in the response variable that can be attributed to variation in the explanatory variables, linear regression analysis can be applied to quantify the strength of the relationship between the response and the explanatory variables, and in particular to determine whether some explanatory variables may have no linear relationship with the response at all, or to identify which subsets of explanatory variables may contain redundant information about the response.

</details>



<details><summary><b>What is Logistic Regression, How it works, When to Use ? </b></summary>

 > Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist
 > Logistic Regression is used when the dependent variable(target) is categorical.

</details>


<details><summary><b>What is Support vector machine, How it works, When to Use ? </b></summary>

 > In machine learning, support-vector machines (SVMs, also support-vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall. 

 > The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

</details>


<details><summary><b>What is SVR ( Support vector Regressor ) ? </b></summary>

 > [read more](https://medium.com/coinmonks/support-vector-regression-or-svr-8eb3acf6d0ff)

</details>


<details><summary><b> What is SVC ( Support Vector Classification ) ? </b></summary>

 >

</details>


<details><summary><b> What is KNN( K nearest neighbour algorithm ) ? </b></summary>

 > Knn is a supervised learning algorithm,

</details>


<details><summary><b> How to choose k value in KNN ? </b></summary>

 > sqrt(n) :  n is the number of samples

</details>


<details><summary><b> What is Ecludien distance ? </b></summary>

 > In mathematics, the Euclidean distance or Euclidean metric is the "ordinary" straight-line distance between two points in Euclidean space.

</details>

<details><summary><b>What is Naive bayes algorithm ? How it works ? </b></summary>

 > Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features. 

</details>

<details><summary><b> What is ensemble learning? </b></summary>

 > [answer](https://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/)

</details>


<details><summary><b> What is Decision Tree algorithm ? How the tree will split ?</b></summary>

 > [answer](https://www.geeksforgeeks.org/decision-tree-introduction-example/)

</details>


<details><summary><b> What is Random Forest algorithm ? How to pick no of trees ?</b></summary>

 > It consists of a large number of individual decision trees that operate as an ensemble.

</details>

<details><summary><b>What is Bagging ? </b></summary>

 > Bagging is used typically when you want to reduce the variance while retaining the bias. n bagging, first you will have to sample the input data (with replacement) to generate multiple sets of input data. For each of those sets, the same baseline predictor (such as a SVM, Neural Net, etc) is run to get a trained model for each of the training set.

</details>


<details><summary><b> What is Boosting ? </b></summary>

 > Boosting is a machine learning ensemble meta-algorithm for primarily reducing bias, and also variance[1] in supervised learning, and a family of machine learning algorithms that convert weak learners to strong ones.[2] Boosting is based on the question posed by Kearns and Valiant (1988, 1989):[3][4] "Can a set of weak learners create a single strong learner?" A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification. 

</details>


<details><summary><b> How many Types of boosting algorithms are there ? </b></summary>

 >

    AdaBoost
    Gradient Boosting
    XGBoost
    LogitBoost
    LPBoost
    TotalBoost
    BrownBoost
    MadaBoost
    etc


</details>


<details><summary><b> What is xgboost algorithm ?</b></summary>

 > [answer](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/)

</details>


<details><summary><b>What is Adaboost Algorithm ? </b></summary>

 >

</details>

<details><summary><b> What is Gradient Boosting algorithm ? </b></summary>

 >

</details>

<details><summary><b>How Gradient Boosting helps to optimize the cost function. </b></summary>

 >

</details>

<details><summary><b>What is Time series ? </b></summary>

 >

</details>

<details><summary><b> How many types of algorithms in time series ? </b></summary>

 >

</details>

<details><summary><b> What is ARIMA Model(Auto regressive and Moving average ) ? </b></summary>

 >

</details>
v

<details><summary><b> What is Customer segmentation ? How can do it with Machine learning ?</b></summary>

 >

</details>

<details><summary><b> What is K- Means ? </b></summary>

 > K means Clustering is unsupervised algorithm to determine the best possible clusters from the data.  The goal of the algorithm to find groups with in data.

</details>

<details><summary><b> How choose K value in Kmeans algorithm ?</b></summary>

 > We can use the elbow method to determine the optimal number of clusters( Kvalue)



</details>

<details><summary><b> How many types of clustering techniques are there ? </b></summary>

 > 
    Partitioning methods.
    Hierarchical clustering.
    Fuzzy clustering.
    Density-based clustering.
    Model-based clustering.


</details>


<details><summary><b>What is Hierarchical clustering ?  </b></summary>

 >

</details>


<details><summary><b>What is Dimentionality Reduction ? How it works ?  </b></summary>

 >

</details>

<details><summary><b> How many types dimentionality reduction techniques ? </b></summary>

 >

</details>


<details><summary><b> What is PCA ? ( Principal component analysis ) </b></summary>

 > Principal component analysis is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. It’s mainly used to reduce dimentionality of data set.

</details>


<details><summary><b> What type of metrics in Regression ? </b></summary>

 >RMSE – Root Mean square error

> MSE – Mean square error

> MAE – Mean absolute Error

> R2 score

</details>


<details><summary><b>How to improve the model accuracy ? </b></summary>

 > By using Feature selection, Dimensionality reduction, Ensemble methods(bagging and boosting algorithms) and Hyper parameter tuning.

</details>


<details><summary><b> How many types of loss function or cost function in machine learning ?  </b></summary>

 > Classification:

    log loss
    focal loss
    KL Divergence/Relative entropy
    Exponential loss
    Hinge Loss

> Regression:

    mean square error
    mean absolute error
    huber loss/ smooth mean absolute error
    log cosh loss
    quantile loss


</details>


<details><summary><b> Which one you prefer model performance an model accuracy while building model ?</b></summary>

 > I can use model performance, model accuracy is the subset of model performance.

</details>


<details><summary><b> What is Mean square error , formula and criteria ?</b></summary>

 >

</details>


<details><summary><b>What is Root Mean Square error ? </b></summary>

 >

</details>

<details><summary><b> What is  R2 score. ?</b></summary>

 >

</details>

<details><summary><b>What type of metrics in Classification ? </b></summary>

 >
    Confusion Matrix = ((TP + FN)/(FP + TN))
    Accuracy score = (TP+TN)/TP+TN+FP+FN
    Recall , True positive rate, – ( TP/TP+FN)
    Precision – (TP/TP+TN)
    F1score = 2(precision*recall)/precision+recall


</details>

<details><summary><b> How can you overcome from overfitting ?</b></summary>

 >

</details>
<details><summary><b> How can you overcome from underfitting ?</b></summary>

 >

</details>

<details><summary><b>What is Meant by normalization ? </b></summary>

 >

</details>


<details><summary><b>What is meant by dummy variables ? </b></summary>

 >

</details>

<details><summary><b>What is Regularization ? </b></summary>

 >

</details>



<details><summary><b>What is Different L1 Regularization and L2 Regularization ? </b></summary>

 >

</details>
<details><summary><b>How can you deal with different types of seasonality in time series modelling ? </b></summary>

 >

</details>

<details><summary><b>What is Multicollinerity ? </b></summary>

 >

</details>

<details><summary><b> What is ROC Curve ?</b></summary>

 >

</details>

<details><summary><b>What is Sigmod Function ? </b></summary>

 >

</details>

<details><summary><b> Which one i have to learn for Data science Python or R programming language ?</b></summary>

 >

</details>

<details><summary><b>What is Data visualization with different Charts in Python ? </b></summary>

 1. Histogram,

2. Bar plots

3. Linegraph

4. Pie Chart

5. Scatter Plot

6. Box plots

</details>

<details><summary><b>What is best programming libraries of machine learning. </b></summary>

 > R, Python, numpy, scikitlearn, pandas, Scikit Learn, Tensorflow, Keras, Pytorch, Matplotlib, Seaborn

</details>

##### Programming

<details><summary><b> What is the current version of python ?</b></summary>

 >

</details>


<details><summary><b> Why Python for data science ?</b></summary>

 >

</details>


<details><summary><b> What is difference between lists and tuples ?</b></summary>

 >

</details>


<details><summary><b>How can do webscaping in Python ? </b></summary>

 >

</details>


<details><summary><b>What are libraries in python ? </b></summary>

 > 
    Numpy
    Pandas
    Scipy
    Scikit Learn
    Tensorflow
    Keras
    Pytorch
    Matplotlib
    Seaborn


</details>


<details><summary><b>What is scikit learn library ? </b></summary>

 >

</details>

<details><summary><b> What is scipy library ?</b></summary>

 >

</details>

##### Numpy Interview questions and Answers

<details><summary><b> What is Numpy Arrays ?</b></summary>

 >

</details>



<details><summary><b>What is different between numpy and arrays ? </b></summary>

 >

</details>

##### Pandas Interview Questions  and Answers

<details><summary><b>What is Pandas ? </b></summary>

 >

</details>


<details><summary><b>What is use of Pandas ? </b></summary>

 >

</details>


<details><summary><b> How to find duplicate values and remove in data  by using pandas?</b></summary>

 >

</details>


<details><summary><b>How to find null values  in data by using pandas ? </b></summary>

 >

</details>


<details><summary><b> How to sort data using pandas ?</b></summary>

 >

</details>

<details><summary><b>How you can fill null values ? </b></summary>

 >

</details>

<details><summary><b> How to convert string to date object ?</b></summary>

 >

</details>

##### Deep Learning Interview Questions

<details><summary><b> What is Neural Network ?</b></summary>

 >

</details>

<details><summary><b>Types of Neural Network ? </b></summary>

 >

</details>


<details><summary><b> What is MLP ?</b></summary>

 >

</details>


<details><summary><b> What is CNN ?v</b></summary>

 >

</details>



<details><summary><b>What Is RNN ? </b></summary>

 >

</details>


<details><summary><b>What is LSTM ? </b></summary>

 >

</details>


<details><summary><b>What is GRU ? </b></summary>

 >

</details>



<details><summary><b> Why LSTM is better than Recurrent Neural Network  ?</b></summary>

 >

</details>


<details><summary><b>What are encoders ? </b></summary>

 >

</details>



<details><summary><b>What is GANS ? </b></summary>

 >

</details>



<details><summary><b>What is Deep Belief Network ? </b></summary>

 >

</details>



<details><summary><b>What is Activation function ? </b></summary>

 >

</details>


<details><summary><b>How many types of Activation Functions are there ? </b></summary>

 >

</details>

<details><summary><b>What is Dropout in NN ? </b></summary>

 >

</details>

<details><summary><b>What are unsupervised Learning Algorithms in deep learning . </b></summary>

 >

</details>

##### Natural Language Processing  Interview questions. 

<details><summary><b>What NLP ? </b></summary>

 >

</details>

<details><summary><b> What is tokenizing ?</b></summary>

 >

</details>
<details><summary><b> What is Stemming ? </b></summary>

 >

</details>
<details><summary><b> What is Lemmatizing ? </b></summary>

 >

</details>

<details><summary><b> What is POS Tagging ?</b></summary>

 >

</details>

<details><summary><b>What is Genism ? </b></summary>

 >

</details>

<details><summary><b> What is Word2Vec Model ?</b></summary>

 >

</details>

<details><summary><b> What is BiGram, Trigram ? </b></summary>

 >

</details>



<details><summary><b>What are some applications of Machine learning ? </b></summary>

 >

</details>

<details><summary><b>When to use deep learning methods ? </b></summary>

 >

</details>

##### Tensorflow Interview questions and Answers:
<details><summary><b>What is tensorflow ? </b></summary>

 >

</details>

<details><summary><b>What is tensor ? </b></summary>

 >

</details>

<details><summary><b>What is session ? </b></summary>

 >

</details>


<details><summary><b>What is constant in tensorflow ? </b></summary>

 >

</details>

<details><summary><b>What is tensorboard ? </b></summary>

 >

</details>

#####  Tableau Interview Questions:


<details><summary><b>What is Tableau ? </b></summary>

 >

</details>



<details><summary><b>What is difference between tableau and power BI ? </b></summary>

 >

</details>

##### SQl inteview Questions:

<details><summary><b>What is Sql ? </b></summary>

 >

</details>


<details><summary><b>Type of Joins in SQL ? </b></summary>

 >

</details>

<details><summary><b> Types of Clauses in SQL ?</b></summary>

 >

</details>




###### [dezyre.com](https://www.dezyre.com/article/data-analyst-interview-questions-to-prepare-for-in-2018/324)


<details><summary><b>What is the difference between Data Mining and Data Analysis? </b></summary>

 > Data Mining
     Data mining usually does not require any hypothesis. 	
     Data Mining depends on clean and well-documented data. 
     Results of data mining are not always easy to interpret. 	
     Data mining algorithms automatically develop equations.

 > Data Analysis
    Data analysis begins with a question or an assumption.
    Data analysis involves data cleaning.
    Data analysts interpret the results and convey the to the stakeholders.
	Data analysts have to develop their own equations based on the hypothesis.

</details>


<details><summary><b>Explain the typical data analysis process. </b></summary>

 > Data analysis deals with collecting, inspecting, cleansing, transforming and modelling data to glean valuable insights and support better decision making in an organization. The various steps involved in the data analysis process include –

 > Data Exploration –

    Having identified the business problem, a data analyst has to go through the data provided by the client to analyse the root cause of the problem.

 > Data Preparation

    This is the most crucial step of the data analysis process wherein any data anomalies (like missing values or detecting outliers) with the data have to be modelled in the right direction.

 > Data Modelling

    The modelling step begins once the data has been prepared. Modelling is an iterative process wherein the model is run repeatedly for improvements. Data modelling ensures that the best possible result is found for a given business problem.

 > Validation

    In this step, the model provided by the client and the model developed by the data analyst are validated against each other to find out if the developed model will meet the business requirements.

 > Implementation of the Model and Tracking

    This is the final step of the data analysis process wherein the model is implemented in production and is tested for accuracy and efficiency.

</details>


<details><summary><b> What is the difference between Data Mining and Data Profiling?</b></summary>

 > Data Profiling, also referred to as Data Archeology is the process of assessing the data values in a given dataset for uniqueness, consistency and logic. Data profiling cannot identify any incorrect or inaccurate data but can detect only business rules violations or anomalies. The main purpose of data profiling is to find out if the existing data can be used for various other purposes.

 Data Mining refers to the analysis of datasets to find relationships that have not been discovered earlier. It focusses on sequenced discoveries or identifying dependencies, bulk analysis, finding various types of attributes, etc.

</details>


<details><summary><b> How often should you retrain a data model? </b></summary>

 > A good data analyst is the one who understands how changing business dynamics will affect the efficiency of a predictive model. You must be a valuable consultant who can use analytical skills and business acumen to find the root cause of business problems.

 > The best way to answer this question would be to say that you would work with the client to define a time period in advance. However, I would refresh or retrain a model when the company enters a new market, consummate an acquisition or is facing emerging competition. As a data analyst, I would retrain the model as quick as possible to adjust with the changing behaviour of customers or change in market conditions.

</details>


<details><summary><b> What is data cleansing? Mention few best practices that you have followed while data cleansing. </b></summary>

 > From a given dataset for analysis, it is extremely important to sort the information required for data analysis. Data cleaning is a crucial step in the analysis process wherein data is inspected to find any anomalies, remove repetitive data, eliminate any incorrect information, etc. Data cleansing does not involve deleting any existing information from the database, it just enhances the quality of data so that it can be used for analysis.

</details>


<details><summary><b> How will you handle the QA process when developing a predictive model to forecast customer churn? </b></summary>

 > Data analysts require inputs from the business owners and a collaborative environment to operationalize analytics. To create and deploy predictive models in production there should be an effective, efficient and repeatable process. Without taking feedback from the business owner, the model will just be a one-and-done model.

 The best way to answer this question would be to say that you would first partition the data into 3 different sets Training, Testing and Validation. You would then show the results of the validation set to the business owner by eliminating biases from the first 2 sets. The input from the business owner or the client will give you an idea on whether you model predicts customer churn with accuracy and provides desired results.

</details>



<details><summary><b> Mention some common problems that data analysts encounter during analysis.</b></summary>

 > 
    Having a poor formatted data file. For instance, having CSV data with un-escaped newlines and commas in columns.
    Having inconsistent and incomplete data can be frustrating.
    Common Misspelling and Duplicate entries are a common data quality problem that most of the data analysts face.
    Having different value representations and misclassified data.


</details>


<details><summary><b> What are the important steps in data validation process? </b></summary>

 > Data Validation is performed in 2 different steps-

 > Data Screening – In this step various algorithms are used to screen the entire data to find any erroneous or questionable values. Such values need to be examined and should be handled.

 > Data Verification- In this step each suspect value is evaluated on case by case basis and a decision is to be made if the values have to be accepted as valid or if the values have to be rejected as invalid or if they have to be replaced with some redundant values.

</details>


<details><summary><b>How will you create a classification to identify key customer trends in unstructured data? </b></summary>

 > A model does not hold any value if it cannot produce actionable results, an experienced data analyst will have a varying strategy based on the type of data being analysed. For example, if a customer complain was retweeted then should that data be included or not. Also, any sensitive data of the customer needs to be protected, so it is also advisable to consult with the stakeholder to ensure that you are following all the compliance regulations of the organization and disclosure laws, if any.

 You can answer this question by stating that you would first consult with the stakeholder of the business to understand the objective of classifying this data. Then, you would use an iterative process by pulling new data samples and modifying the model accordingly and evaluating it for accuracy. You can mention that you would follow a basic process of mapping the data, creating an algorithm, mining the data, visualizing it and so on. However, you would accomplish this in multiple segments by considering the feedback from stakeholders to ensure that you develop an enriching model that can produce actionable results.

</details>



<details><summary><b> What is the criteria to say whether a developed data model is good or not?</b></summary>

 > 
    The developed model should have predictable performance.
    A good data model can adapt easily to any changes in business requirements.
    Any major data changes in a good data model should be scalable.
    A good data model is one that can be easily consumed for actionable results.


</details>


<details><summary><b> According to you what are the qualities/skills that a data analyst must posses to be successful at this position. </b></summary>

 > Problem Solving and Analytical thinking are the two important skills to be successful as a data analyst. One needs to skilled ar formatting data so that the gleaned information is available in a easy-to-read manner. Not to forget technical proficiency is of significant importance. You can also talk about other skills that the interviewer expects in an ideal candidate for the job position based on the given job description.

</details>

<details><summary><b>You are assigned a new data anlytics project. How will you begin with and what are the steps you will follow? </b></summary>

 > The purpose of asking this question is that the interviewer wants to understand how you approach a given data problem and what is the though process you follow to ensure that you are organized. You can start answering this question by saying that you will start with finding the objective of the given problem and defining it so that there is solid direction on what need to be done. The next step would be to do data exploration and familiarise myself with the entire dataset which is very important when working with a new dataset.The next step would be to prepare the data for modelling which would including finding outliers, handling missing values and validating the data. Having validated the data, I will start data modelling untill I discover any meaningfuk insights. After this the final step would be to implement the model and track the output results.

 This is the generic data analysis process that we have explained in this answer, however, the answer to your  question might slightly change based on the kind of data problem and the tools available at hand.

</details>


<details><summary><b>What do you know about  interquartile range as data analyst? </b></summary>

 > A measure of the dispersion of data that is shown in a box plot is referred to as the interquartile range. It is the difference between the upper and the lower quartile.

</details>


<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>


<details><summary><b> </b></summary>

 >

</details>

<details><summary><b> </b></summary>

 >

</details>