"""
@package shapes
@module classifiers
@author Anna Schneider
Contains classes for Classifier, SVMClassifier, and GMMClassifier
"""
# import from standard library
from exceptions import ValueError

# import external packages
import numpy as np
from scipy import stats
#from sklearn import svm, mixture

class Classifier:
    def __init__(self, cutoff=0):
        """ Constuctor
        
        @param cutoff Float to set a minimum match value for a data point
            to be counted as a class member (default 0)
        """
        self._rejection_cutoff = cutoff
    
    def compute_match(self, class_data, test_data):
        """ Return value of match of a test data point to a class.
            Larger number means better match.
        
        @param self The object pointer
        
        @param class_data Object holding data to describe class
        
        @param test_data Object holding data to describe test data point

        @retval float >= 0

        """            
        # compute match if test data is valid
        if test_data.is_valid:
            
            # extract features to compare to class
            subshape = test_data.subset(class_data.feature_names)
            test_vals = np.asarray(subshape.vals)
            
            # return match
            return self._match(class_data, test_vals)
            
        # if not, match is 0
        else:
            return 0

    def is_match(self, class_data, match_val):
        """ Return true if value of crystal match is above cutoff

        @param self The object pointer
        
        @param class_data Object holding data to describe class
        
        @param match_val Float for goodness of fit
            
        @retval Bool True if above cutoff (ie matches class),
            False if below cutoff (ie doesn't match class)
        """
        # use class_data's cutoff if it exists
        if hasattr(class_data, "cutoff"):
            cutoff = class_data.cutoff
        else:
            cutoff = self._rejection_cutoff
            
        # compare value to cutoff
        if match_val > cutoff:
            return True
        else:
            return False        
            
    def _match(self, class_rep, test_arr):
        """ Calculate the match of the test features to the class
            representative, a float >=0.
            Should be implemented in every derived class.
            
        @param self The object pointer
        
        @param class_rep Object that describes class to match to
        
        @param test_arr ndarray of features
        """
        return 0

class SVMClassifier(Classifier):        
    def _match(self, class_data, test_arr):
        """ Calculate the match of the test features to the class
            representative, a float >=0.
            Projects features onto the axes of class_rep via
                (class_rep.vals) (dot product) (test_features_vector)
                    + (class_rep.intercept)
            
        @param self The object pointer

        @param class_data Object with attributes vals and intercept
        
        @param test_arr ndarray of features

        """        
        # get vector of vals for class
        class_arr = np.asarray(class_data.vals)
        
        # test that class and test are comparable
        if class_arr.shape != test_arr.shape:
            raise ValueError('class and test feature vectors must have same length')

        # calculate
        match_val = np.dot(class_arr, test_arr) + class_data.intercept
        
        # return
        return match_val
            
class GMMClassifier(Classifier):
    def _match(self, class_data, test_arr):
        """ Calculate the match of the test features to the class
            representative, a float between 0 and 1.
            Calculates the probability of the test vector in the
                multivariate Gaussian described by the class.
            
        @param self The object pointer

        @param class_data Object with attributes vals and sds
        
        @param test_arr ndarray of features

        """
        # get vector of means and sds for class
        class_means = np.asarray(class_data.vals)
        class_sds = np.asarray(class_data.sds)
        
        # test that class and test are comparable
        if class_means.shape != test_arr.shape:
            raise ValueError('class and test feature vectors must have same length')

        # calculate
        prob = 1.0
        for ifeat in class_means.shape[0]:
            prob *= stats.norm.pdf(test_arr[ifeat],
                                   loc=class_means[ifeat],
                                   scale=class_sds[ifeat])
                                   
        # return
        return prob                       
       
        