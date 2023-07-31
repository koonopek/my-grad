# Lesson 1



# Lesson 2

### Notes
* Smoothing used in bigram/trigam model by adding 1 to all counts to make model more general (prevents overfitting to current data set)
* Regularization some effect as Smoothing by achieved in different way, instead of editing data before training (sometimes can't be done in case of NN) we add to loss function something like `normal_cost_fn()+0.01*(W**2).mean()`
Thus we penalize to big weights which may lead to overfitting.
* Cost function to calculate models based on probability:
  * If the model assigns higher probabilities then it means, that it better knows the world, because he is confident in his moves. So in this can we can assign cost function as product of probabilities on given data set.
    * Example: `p(first_char, second_char) => probability that after first_char the next will be second_char` then we just iterative with function like this over data set and multiple probabilities.
    * However multiplying so many numbers (which are less then <0,1>) would let to very small number. Thus we `log` function to turn it into more friendly numbers, and from the property of log we can use sum instead of product. Because `log(a * b) = log(a) + log(b)`. And because log(<0,1>) < 0 we negate it
    * Name: `Maximum likelihood`

### Tasks
* E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
  * For bigrams cost function almost doesn't change => it (may) means that model is generic and it doesn't overfitting on particular data set
  * For trigrams cost function increases (2.0 -> 2.5) it (may) means that trigram models are more complicated thus easier overfitting

* E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss.
  What patterns can you see in the train and dev set loss as you tune this strength? 
  Take the best setting of the smoothing and evaluate on the test set once and at the end. => 
  How good of a loss do you achieve?
  * When we increase the strength of smoothing (or regularization) the model because more generic thus `cost(training) - cost(dev)` decreases, but in this case overall cost(training) also increases significantly, so it makes it NOT worth it.