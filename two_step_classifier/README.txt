These files implement the two-layer classifier described by Shen at al. in "Large-scale Item Categorization for E-Commerce".
See here: https://pdfs.semanticscholar.org/5fdd/245a4a2a0d0010085936586530d1f31f633f.pdf

We apply it to a large dataset of comments from the website Reddit, and attempt to classify its "subreddit". 
This is a similar endeavor to topic modeling.

Essentially, the two-layer classifier works by creating clusters of output classes.
A first-layer classifier is used to classify the cluster membership of a comment. 
There are k second-layer classifiers for a model with k clusters. 
Each second-layer classifier (unique to each of the k clusters) outputs classifications only over the subreddits contained in its respective cluster.

This directory contains the following files:

clustering.py - Develops clusters of Reddit discussion boards
- Organizes Reddit discussion boards as documents containing concatenated comments. 
- Extracts "most influential" (by TF-IDF) tokens for each Reddit
- Uses these tokens as input to a clustering algorithm

hard_two_layer_classifier.py
- Implements the two-layer architecture described in the paper

soft_two_layer_classifier.py 
- Implements a modification over the two layer architecture, where both classifiers output a distribution rather than a classification
- Outputs the five most likely labels across all clusters 
