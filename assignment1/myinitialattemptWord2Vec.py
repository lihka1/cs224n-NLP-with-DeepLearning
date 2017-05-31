#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    x = x.astype(float)
    sums = np.sqrt(np.sum(np.multiply(x,x), axis=1))
    sums = np.reshape(sums, (len(sums),1))
    x = np.divide(x,sums)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    yhat = softmax(np.dot(outputVectors, predicted))
    cost = -np.log(yhat[target]) 
    yhatminusy = yhat
    yhatminusy[target] = yhatminusy[target] - 1
                         
    gradPred = np.dot(outputVectors.T,yhatminusy) 
    grad = np.dot(np.reshape(yhatminusy,(len(yhatminusy),1)),
                  np.reshape(predicted,(1,len(predicted))))
    
    ### END YOUR CODE
    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    
    samples = np.array([outputVectors[ind] for ind in indices[1:]])
    
    #print samples[0]
    #print samples,samples.shape
    UkVc = -np.dot(samples,predicted)
    sig_UkVc = sigmoid(UkVc)
    #print "check,,,,,,,,",sig_UkVc
    samples_cost = -np.sum(np.log(sig_UkVc))
    similar = sigmoid(np.dot(outputVectors[target],predicted))
    #print "check,,,,,,,,,,,,,,",outputVectors[target]
    target_cost = -(np.log(similar))
    cost = target_cost+samples_cost

    #print samples.shape,outputVectors[target],predicted
    
    
    ######## my initial code ..samples can be vectorized i think..lets try
    #negSamplecost = 0
    
    #for i in range(K):
        #negSamplecost += np.log(sigmoid(-np.dot(outputVectors[indices[i+1]],
    #                                             predicted)))
    #cost = -np.log(similar) - negSamplecost
    #print "check,,,,,,,,,,,,,",cost,cost1
    #######################################cost part vectorised
    non_neg_gradPred = (similar-1)*outputVectors[target]
    neg_gradPred = 0
    pro = sigmoid(UkVc)-1
    neg_gradPred = np.sum((np.reshape(pro,(len(pro),1)) *samples),axis=0)
    #k = sigmoid(-np.dot(outputVectors[indices[1]],predicted)) - 1 
    #print outputVectors[indices[1]],k
    #print outputVectors[indices[1]]*k
    #print (sigmoid(-np.dot(outputVectors[indices[1]],predicted)) - 1) * outputVectors[indices[1]]
    #for i in range(K):
      #  neg_gradPred += (sigmoid(-np.dot(outputVectors[indices[i+1]],
         #                                        predicted)) - 1) * outputVectors[indices[i+1]]
    #print "aaaaaaaaaaaaaaaa",outputVectors[0].shape
    gradPred = non_neg_gradPred - neg_gradPred
    #gradPred1 = non_neg_gradPred - neg_gradPred1
    #print "OLAAAAAAAAAAa",gradPred,gradPred1
    
    
    #########################################
    
    #### THIS TOO CAN BE VECTORIZED I THINK....
    # get the output vectors row-wise as in the cross-entropy one
    grad_tar_sample = np.zeros(outputVectors.shape)
    grad_tar_sample1 = np.zeros(outputVectors.shape)
    
    #print "heeeeeeeeeeeeee",np.reshape(predicted,(len(predicted),1))
    #re_predicted = np.reshape(predicted,(len(predicted),1))
    
    #print "hereeeeeeeeeeeeeeeee",pro.shape,outputVectors.shape,predicted.shape
    grad1 = -(np.reshape(pro,(len(pro),1))*predicted)
    
    #print "FIgggggggggggg", -(sigmoid(-np.dot(utarget_sample,predicted))-1) * predicted
    #print grad1[K-1]
    
    #print grad1[5].shape
    for i in range(1,K+1):
        grad_tar_sample1[indices[i]] += grad1[i-1]
    
    
    #for i in range(K+1):
       # utarget_sample = outputVectors[indices[i]]
        #print utarget_samples.shape,utarget_samples.T.shape,re_predicted.shape
     #   grad_tar_sample[indices[i]]  += -(sigmoid(-np.dot(utarget_sample,
      #                                         predicted))-1) * predicted
    

    #print "hiiiiii",grad_tar_samples1.shape,re_predicted.shape,np.reshape(grad_tar_samples1,(len(grad_tar_samples1),1)).shape
    
    #grad_tar_sample = grad_tar_samples1 * predicted
    #print "HOOOOOOOO",grad_tar_sample.shape
    #grad_tar_sample[target] = grad_tar_sample[target] - predicted
    grad_tar_sample1[target] += (similar-1.0)*predicted    

    #grad = grad_tar_sample
    grad = grad_tar_sample1
    ### END YOUR CODE
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    #print outputVectors.shape

    ### YOUR CODE HERE
    #print gradIn
    #print gradOut
    #print C,currentWord
    #print contextWords
    #print tokens
    #print inputVectors
    #print outputVectors    
    for word in contextWords:
        cost1,grad1,grad2 = word2vecCostAndGradient(inputVectors[tokens[currentWord]],
                                           tokens[word],outputVectors,dataset)
#        if i == 0:
#            print "CHECKKKKKKKK"
#            print grad1
#            print gradIn
#            print grad2
        cost += cost1
        gradIn[tokens[currentWord]] += grad1
        gradOut += grad2
            
    
    #raise NotImplementedError
    ### END YOUR CODE
    
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
     #   skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
      #  dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()