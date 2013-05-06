"""
@package classifiers
@author Anna Schneider
@version 0.1
@brief Contains classes for Classifier, SVMClassifier, GMMClassifier,
    and factory method classifier_factory
"""

# import from standard library
import math

# import external packages
import numpy as np
from scipy import stats

class Classifier:
    """ Base class for classifiers. Implement _match to derive a useful
        classifier.
        
    """
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
        
        @param class_data Shape-like object holding data to describe class
        
        @param test_data Shape-like object holding data to describe test data point

        @retval float >= 0

        """            
        # compute match if test data is valid
        if test_data.get('is_valid'):
            
            # extract features to compare to class
            subshape = test_data.subset(class_data.get_components())
            test_vals = np.asarray(subshape.get_vals())
            
            # return match
            return self._match(class_data, test_vals)
            
        # if not, match is 0
        else:
            return 0

    def is_match(self, class_data, match_val):
        """ Return true if value of crystal match is above cutoff

        @param self The object pointer
        
        @param class_data Shape-like object holding data to describe class
        
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
        print cutoff
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
    """ Classifier that implements a support vector machine.
        Calculates the match of the test features to the class
            representative, a float >=0.
        Projects features onto the axes of class_rep via
            (class_rep.vals) (dot product) (test_features_vector)
                + (class_rep.intercept)
        """            
    def _match(self, class_data, test_arr):
        """ Calculates the match.
        
        @param self The object pointer

        @param class_data Object with attributes vals and intercept
        
        @param test_arr ndarray of features

        """        
        # get vector of vals for class
        class_arr = np.asarray(class_data.get_vals())
        
        # test that class and test are comparable
        if class_arr.shape != test_arr.shape:
            raise ValueError('class and test feature vectors must have same length')

        # calculate
        match_val = np.dot(class_arr, test_arr) + class_data.get('intercept')
        
        # return
        return match_val
            
class GMMClassifier(Classifier):
    """ Classifier that implements a Gaussian mixture model with a maximum
        likelihood decision rule.

        Calculates the match of the test features to the class
            representative, a float between 0 and 1.
        Calculates the probability of the test vector in the
            multivariate Gaussian described by the class.
                    
    """
    def _match(self, class_data, test_arr):
        """ Calculates the match.
        
        @param self The object pointer

        @param class_data Object with attributes vals and sds
        
        @param test_arr ndarray of features

        """
        # get vector of means and sds for class
        class_means = np.asarray(class_data.get_vals(), dtype=float)
        class_sds = np.asarray(class_data.get('sds'), dtype=float)
        
        # test that class and test are comparable
        if class_means.shape != test_arr.shape:
            raise ValueError('class and test feature vectors must have same length')

        # calculate
        prob = 1.0
        for ifeat in range(class_means.shape[0]):
            prob *= stats.norm.pdf(test_arr[ifeat],
                                   loc=class_means[ifeat],
                                   scale=class_sds[ifeat])
            
        # return
        return prob                       
       
       
def classifier_factory(classifier_type, cutoff=None):
    """ Function to create a classifier. Valid types must contain the substrings
        'GMM' (for Gaussian Mixture Model) or
        'SVM' (for Support Vector Machine); case insensitive.
    
    @param classifier_type String specifying a valid classifier type
    
    @param cutoff Number for rejection cutoff; valid matches must have a
        score above this value
        
    @retval Classifier object
    
    """
    lower_case_str = classifier_type.lower()
    
    if 'gmm' in lower_case_str:
        return GMMClassifier(cutoff)
    elif 'svm' in lower_case_str:
        return SVMClassifier(cutoff)
    else:
        raise ValueError('invalid classifier type %s' % classifier_type)