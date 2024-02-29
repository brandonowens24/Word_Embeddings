# Report
## Embeddings
**Model 1**: Continuous-Bag of Words Word2Vec Model
* This model used the gensim library, created an embedding vector of 300 values, used a window of 3, hierarchical softmax, and trained over 50 epochs.
* I used the sg=0 argument so it took on the CBOW format. These models try to predict a word from its context.
<br></br>

**Model 2**: Skip-Gram Word2Vec Model

* This model used the gensim library, created an embedding vector of 300 values, used a window of 3, hierarchical softmax, and trained over 50 epochs.
* I used the sg=1 argument so it took on the SG format. These models try to predict context given a word.
<br></br>

**Programming**: <br></br>
These models were extremely easy to program(after I read the documentation of course). They are very similarly created using the gensim library, which only required me to change a singular argument to
differentiate between the model type. At the same time however, they took a good chuck of time to train. It wasn't so bad after I saved and downloaded them to use in the future, but initial training was lengthy.
<br></br>

**Comparison to GloVe and Google Word Embeddings**

**Singular Most Similar Words**
  
All four models predicted the most similar word to the word `calculator` was "calculators", but after that similarity they began to differ. The Google model took the approach of capitalization, while the CBOW
and SG Model liked the word "computer" and "device". This was similar to GloVe that came up with "computerized". Perhaps this shows that my models were a bit better semantically with singular words as Google looks
to have clinged onto a more lexical approach.
<br></br>

**Vector Arithmetic**
  
All four models were able to predict that `("faster" - "fast" + "strong =)`'s top result was stronger (which was what I was looking for). At the same time, I used the arithmetic `("england" - "london" + "madrid" = )`
and all the models came up with "spain"... except for the GloVe model. Spain is an answer that makes sense because as London is the capital of England, Madrid is the capital of Spain. This may be due to the fact
that the GloVe model I chose only has 50 values for each word embedding. Perhaps this is not enough features for finding similar vector distance geographically.
<br></br>

**Odd One Out**
  
All four models were able to correctly identify that the word "candle" did not fit in with the list `("finance", "business", "market", "candle", "stock")`.
<br></br>

**Sentence Similarity**
  
Here I compared similarity scores of the sentences: `"I picked you up from the airport"` and `"I got you from your flight yesterday"` -- which semantically, are similar sentences. They both denote that the subject,
I collected you from your air travel at some point in the past. With that being said, the GloVe and Google models performed extremely well (0.94, 0.82 similarity scores, respectively). Meanwhile,
my models didn't fare terribly, but not nearly as well as the GloVe and Google. CBOW had 0.43 and SG had 0.53. It makes sense that the skip-gram model was better at determining similarity of context
because it computes its features from predicting word contexts.
<br></br>

## Bias
**Bias Topic**

I selected the Word Embeddings Association Test (WEAT) to test for Word Embedding Fairness Evaluation (WEFE). This metric recieves two sets of target words and two sets of attribute words with the
objective of quantifying teh strength and association of both pair of sets through a permutation test. Ideally, a score of 0 infers no bias. 
I chose to try to focus if there was any racial bias associated with the words `black` or `hispanic` compared to that of `white` and `caucasian`.
<br></br>

Surprisingly, the models showed effect scores of:
* CBOW: -0.1
* SG: -0.04
* GloVe: -.0.06
* Google: 0.03
<br></br>

Meaning that none of the four models showed bias with the attributes I selected. With that being said, there can be some definite problems here: perhaps the attributes I selected were poorly chosen
or had double meanings, perhaps my targets words didn't classify enough of a target, or maybe online text sources have done a better job at eliminating underlying racism with writing. I imagine that 
if the topic I chose was different with different targets or even just different attributes and bias was found, this could cause a machine learning model to incorrectly associate specific words with
specific, commonly associated contexts that resemble implicit biases in text like racism, sexism, etc. With that association, models might deliver misinformation or demonstrate an offensive bias to potential users.
<br></br>

## Classification
I chose an imdb dataset that looked at polarized reviews and their associated sentiment. Then created a BOW LR and a CBOW LR model that used the averages from my CBOW word2vec model from earlier's text 
embeddings and looked at which words appeared in that review. It then averaged all of these embedded vectors for each word in the review and ended up with a 300 feature vector for each review. This was then put through
the same logistic regression as the normal BOW model.
<br></br>
In terms of metrics, I think `accuracy` is a fair metric in this standard. I am primarily interested with how many correct predictions were found between the negative and positive labels.
<br></br>

**Simple BOW Logistic Regression**
* 87% Accurate on Test Data
  
**CBOW Embedded Features Logistic Regression**
* 79% Accurate on Test Data

**Why These Results**: 
I believe in this case, the way the CBOW LR was formed, I analyzed each token in a review and assigned it my CBOW vectors from earlier, then averaged all of those vectors out. This is problematic for two reasons:
* The CBOW model was simple: Only 50 epochs, no aggregate hyperparameter testing, lacked sophistication
* The reviews may have contained words that weren't vectors in the CBOW model and were instead ignored
Both of these problems would hurt the accuracy of the CBOW Embedded Features Logistic Regression model: thus, in this case, the simple model worked better.

## Reflection
Besides learning how awesome and easy to use gensim is, I was able to get experience with Google Colab and Jupyter notebook. I have used these tools before, but have never actually built and stored models with them before.
This was particularly challenging, as I forgot to download my model the first time and had to wait around 5 hours for training to use it in the future. Luckily I learned that gensim lets you save at certain Epochs.
Additionally, I learned the difference between skip-gram and bag-of-words word embedding models and that there are publicly available models on the internet from awesome text sources like Google and Wikipedia. In the future,
I would like to play with these models more an tune some hyper-parameters for them to try to get models as strong as the google and GloVe model (which worked well considering each embedding was only 50 features large).
