Python Project Template
---

### 1. Prepare Problem  
a) Load libraries  
b) Load dataset  

### 2. Summarize Data  
a) Descriptive statistics  
b) Data visualizations  
 
###  3. Prepare Data  
a) Data Cleaning  
b) Feature Selection  
c) Data Transforms  

###  4. Evaluate Algorithms  
a) Split-out validation dataset  
b) Test options and evaluation metric  
c) Spot Check Algorithms  
d) Compare Algorithms  

### 5. Improve Accuracy  
a) Algorithm Tuning  
b) Ensembles  

###  6. Finalize Model
a) Predictions on validation dataset  
b) Create standalone model on entire training dataset  
c) Save model for later use 

视频课程
---
[How to Win a Data Science Competition: Learn from Top Kagglers](https://www.coursera.org/learn/competitive-data-science/home/welcome)

Blog
---
https://machinelearningmastery.com/blog/

Overview of methods  
---
[Scikit-Learn (or sklearn) library](http://scikit-learn.org/stable/)   
[Overview of k-NN](http://scikit-learn.org/stable/modules/neighbors.html)(sklearn's documentation)  
[Overview of Linear Models](http://scikit-learn.org/stable/modules/linear_model.html)(sklearn's documentation)  
[Overview of Decision Trees](http://scikit-learn.org/stable/modules/tree.html)(sklearn's documentation)  
[Decision Tree](https://en.wikipedia.org/wiki/Decision_tree_learning)(Wiki)  
Overview of algorithms and parameters in [H2O documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)    

Additional Tools  
---
[Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit) repository  
[XGBoost](https://github.com/dmlc/xgboost) repository  
[LightGBM](https://github.com/Microsoft/LightGBM) repository  
[Interactive demo](http://playground.tensorflow.org/) of simple feed-forward Neural Net  
Frameworks for Neural Nets: [Keras](https://keras.io/),[PyTorch](http://pytorch.org/),[TensorFlow](https://www.tensorflow.org/),[MXNet](http://mxnet.io/),[Lasagne](https://lasagne.readthedocs.io/en/latest/)  
[Example from sklearn with different decision surfaces](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)  
[Arbitrary order factorization machines](https://github.com/geffy/tffm)  

knowledge of basic ML algorithms
---
[Explanation of Random Forest](https://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively)  
[Explanation/Demonstration of Gradient Boosting](https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)  
[Example of kNN](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)  

StandCloud Computing
---
[AWS](https://aws.amazon.com/cn/), [Google Cloud](https://cloud.google.com/), [Microsoft Azure](https://azure.microsoft.com/zh-cn/)  

AWS spot option
---
[Overview of Spot mechanism](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
[Spot Setup Guide](https://datasciencebowl.com/aws_guide/)  

Stack and packages
---
[Basic SciPy stack (ipython, numpy, pandas, matplotlib)](https://www.scipy.org/)  
[Jupyter Notebook](https://jupyter.org/)  
[Stand-alone python tSNE package](https://github.com/danielfrg/tsne)  
Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM]()  
Another tree-based method: RGF ([implemetation](https://github.com/baidu/fast_rgf), [paper](https://arxiv.org/pdf/1109.0887.pdf))  
Python distribution with all-included packages: [Anaconda](https://www.continuum.io/what-is-anaconda)  
[Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)  

特征预处理
---
* 数字特征  
* 类别和顺序特征  
* 时间和坐标特征  
* 缺失值  

Order statistic
---
Wikipedia:https://en.wikipedia.org/wiki/Order_statistic  
Scipy:https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.stats.mstats.rankdata.html

Feature preprocessing
---
[Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)  
[Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning)  
[Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)    

Feature Important
---
MARS, Random Forest and Gradient Boosted Machines  

Feature generation
---
[Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)  
[Feature engineering](https://www.quora.com/What-are-some-general-tips-on-feature-selection-and-engineering-that-every-data-scientist-should-know)
[Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)  

Feature extraction
---
* PCA
* Unsupervised clustering methods

Feature selection
---
* [Stepwise regression](https://www.cnblogs.com/sumuncle/p/5647722.html)
* [An Introduction to Feature Selection](https://machinelearningmastery.com/an-introduction-to-feature-selection/)
* Regularization methods LASSO
* [Feature Selection to Improve Accuracy and Decrease Training Time](https://machinelearningmastery.com/feature-selection-to-improve-accuracy-and-decrease-training-time/)
* [Feature Selection in Python with Scikit-Learn](https://machinelearningmastery.com/feature-selection-in-python-with-scikit-learn/)

Feature extraction from text
---
### Bag of words
* [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
* [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/) 

### Word2vec
* [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)  
* [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)  
* [Text Classification With Word2Vec](https://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)  
* [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)  

### NLP Libraries
* [NLTK](http://www.nltk.org/)  
* [TextBlob](https://github.com/sloria/TextBlob)  

Feature extraction from images
---
### Pretrained models
* [Using pretrained models in Keras](https://keras.io/applications/)
* [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)

### Finetuning
* [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
* [Fine-tuning Deep Learning Models in Keras](https://www.tensorflow.org/tutorials/image_retraining)

Visualization tools
---
* [Seaborn](https://seaborn.pydata.org/) 
* [Plotly](https://plot.ly/python/)  
* [Bokeh](https://github.com/bokeh/bokeh)  
* [ggplot](http://ggplot.yhathq.com/)  
* [Graph visualization with NetworkX](https://networkx.github.io/)  
### Others
* [Biclustering algorithms for sorting corrplots](http://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)  


* [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard  
* [Page about data leakages on Kaggle](https://www.kaggle.com/docs)  

Classification
---
* [Evaluation Metrics for Classification Problems: Quick Examples + References](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)  
* [Decision Trees: “Gini” vs. “Entropy” criteria](https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria)  
* [Understanding ROC curves](http://www.navan.name/roc/)  

Ranking
---
* [Learning to Rank using Gradient Descent -- original paper about pairwise method for AUC optimization](https://icml.cc/2015/wp-content/uploads/2015/06/icml_ranking.pdf)  
* [Overview of further developments of RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)  
* [RankLib (implemtations for the 2 papers from above)](https://sourceforge.net/p/lemur/wiki/RankLib/)  
* [Learning to Rank Overview](https://wellecks.wordpress.com/2015/01/15/learning-to-rank-overview/)  

Clustering
---
* [Evaluation metrics for clustering](http://nlp.uned.es/docs/amigo2007a.pdf)  

Hyper-parameters Turning
---
* [Tuning the hyper-parameters of an estimator (sklearn)](http://scikit-learn.org/stable/modules/grid_search.html)  
* Optimizing hyperparameters with hyperopt  
* [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)  

Competitions Tips
---
* [Far0n's framework for Kaggle competitions "kaggletils"](https://github.com/Far0n/kaggletils)  
* [28 Jupyter Notebook tips, tricks and shortcuts](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)  

Matrix Factorization
---
* [Overview of Matrix Decomposition methods (sklearn)](http://scikit-learn.org/stable/modules/decomposition.html)  

t-SNE
---
* [Multicore t-SNE implementation](https://github.com/DmitryUlyanov/Multicore-TSNE)  
* [Comparison of Manifold Learning methods (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html)  
* [How to Use t-SNE Effectively (distill.pub blog)](https://distill.pub/2016/misread-tsne/)  
* [tSNE homepage (Laurens van der Maaten)](https://lvdmaaten.github.io/tsne/)  
* [Example: tSNE with different perplexities (sklearn)](http://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py)  

Interactions
---
* [Facebook Research's paper about extracting categorical features from trees](https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/)  
* [Example: Feature transformations with ensembles of trees (sklearn)](http://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html)  

Ensembling
---
* [Kaggle ensembling guide at MLWave.com (overview of approaches)](https://mlwave.com/kaggle-ensembling-guide/)  
* [Heamy — a set of useful tools for competitive data science (including ensembling)](https://github.com/rushter/heamy)  
* [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)  
* [如何在 Kaggle 首战中进入前 10%](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)
* [Pipeline](https://github.com/ChenglongChen/Kaggle_CrowdFlower)   
* [Beating Kaggle the easy way](http://www.ke.tu-darmstadt.de/lehre/arbeiten/studien/2015/Dong_Ying.pdf)  
* [a Python package for stacking](https://github.com/vecxoz/vecstack)  
* [Machine learning extension package with stacking](https://github.com/rasbt/mlxtend)  
* [Data pipeline package with stacking features](https://github.com/mpearmain/gestalt)  
* [Python helper functions and examples](https://github.com/ndemir/stacking) 

Here are some written in other languages:
* [Python based but built on a Redis backend](https://github.com/reiinakano/xcessiv)  
* [StackNet — a computational, scalable and analytical meta modelling framework (by KazAnova)](https://github.com/kaz-Anova/StackNet)  

Hope this helps!

Past solutions
---
* http://ndres.me/kaggle-past-solutions/  
* https://www.kaggle.com/wiki/PastSolutions  
* http://www.chioka.in/kaggle-competition-solutions/  
* https://github.com/ShuaiW/kaggle-classification/

论文复现
---
* [Matthew Rahtz的博客文章《从复现一篇深度强化学习论文学到的教训》](http://amid.fish/reproducing-deep-rl)
* [S. Zayd Enam的博客文章《为什么机器学习“难”？》](http://ai.stanford.edu/~zayd/why-is-machine-learning-hard.html)  
* [Mendeley](https://www.mendeley.com/)  
* [Paperpile](https://paperpile.com/)  
* [Zotero](https://www.zotero.org/)  
* [Shortscience](http://www.shortscience.org/)  
* [OpenReview](https://openreview.net/)  
* [Scirate](https://scirate.com/)  
* [Papers With Code](https://paperswithcode.com/)  
* [Github pwc](https://github.com/zziz/pwc)  
* [GitXiv](http://www.gitxiv.com/)  
