"""
@package classifiers
@author Anna Schneider
@version 0.1
@brief Contains classes for Classifier, SVMClassifier, GMMClassifier,
    and factory method classifier_factory
"""

# import from standard library
import warnings
import logging

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
        self._rejection_cutoff = float(cutoff)
        
    def algorithm(self):
        """ Returns a string indicating the algorithm used. """
        return 'default'
    
    def compute_match(self, class_data, test_data):
        """ Return value of match of a test data point to a class.
            Larger number means better match.
        
        @param self The object pointer
        
        @param class_data Shape-like object holding data to describe class
        
        @param test_data Shape-like object holding data to describe test data point

        @retval matchval float >= 0

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
            logging.debug('using class cutoff %0.4f' % cutoff)
        else:
            cutoff = self._rejection_cutoff
            logging.debug('using clf cutoff %0.4f' % cutoff)
            
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
        warnings.warn('Getting meaningless classification from default classifier.',
                      RuntimeWarning)
        return 0

class SVMClassifier(Classifier):        
    """ Classifier that implements a support vector machine.
        Calculates the match of the test features to the class
            representative, a float >=0.
        Projects features onto the axes of class_rep via
            (class_rep.vals) (dot product) (test_features_vector)
                + (class_rep.intercept)
        """            
    def algorithm(self):
        """ Returns a string indicating the algorithm used. """
        return 'SVM'

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
        return max(match_val, 0.0)
            
class GMMClassifier(Classifier):
    """ Classifier that implements a Gaussian mixture model with a maximum
        likelihood decision rule.

        Calculates the match of the test features to the class
            representative, a float between 0 and 1.
        Calculates the probability of the test vector in the
            multivariate Gaussian described by the class.
                    
    """
    def algorithm(self):
        """ Returns a string indicating the algorithm used. """
        return 'GMM'

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
       
       
class CARTClassifier(Classifier):
    """ Classifier that implements a classification tree (CART).
        Uses a tree of decision rules to classify shapes.  
        
        Decision rules are stored in the 'rules' property of Shape.
        Format is shape.get('rules') = [('var', val, 'op'), ...]
        where 'var' is the name of a parameter to compare to,
        val is the value to compare to,
        and 'op' is a string indicating the comparison operator.
        The options for operators are ('eq', 'ne', 'gt', 'ge', 'lt', 'le'),
        with the same meanings as in a bash if statement.
        All comparisons must evaluate to True in order for is_match to be True.

    """            
    def algorithm(self):
        """ Returns a string indicating the algorithm used. """
        return 'CART'
   
    def compute_match(self, class_data, test_data):
        """ Return value of match of a test data point to a class.
            1 if all decision rules are satisfied; 0 any rule fails.
                    
        @param self The object pointer
        
        @param class_data Shape-like object holding data to describe class
        
        @param test_data Shape-like object holding data to describe test data point

        @retval matchval float 0 or 1

        """            
        # get vector of decision rules for class
        rules = class_data.get('rules')
        
        # get value of data to test against each rule
        test_vals = np.asarray([test_data.get(rule[0]) for rule in rules])

        # return match
        return self._match(class_data, test_vals)
            
    def _match(self, class_data, test_arr):
        """ Calculates the match.
        
        @param self The object pointer

        @param class_data Object with attributes vals and intercept
        
        @param test_arr ndarray of features

        """        
        # get vector of decision rules for class
        rules = class_data.get('rules')

        # test for failing rules
        for irule, rule in enumerate(rules):
            if rule[2] == 'eq': # equality
                if test_arr[irule] != rule[1]:
                    return 0.0                
            elif rule[2] == 'ne': # inequality
                if test_arr[irule] == rule[1]:
                    return 0.0                
            elif rule[2] == 'gt': # greater than
                if test_arr[irule] <= rule[1]:
                    return 0.0                
            elif rule[2] == 'ge': # greater than or equal to
                if test_arr[irule] < rule[1]:
                    return 0.0                
            elif rule[2] == 'lt': # less than
                if test_arr[irule] >= rule[1]:
                    return 0.0                
            elif rule[2] == 'le': # less than or equal to
                if test_arr[irule] > rule[1]:
                    return 0.0                
            else:
                raise ValueError("Invalid operator %s in decision rule %d" % (rule[2], irule))
                
        # return passing
        return 1.0
       
def classifier_factory(classifier_type='default', cutoff=0):
    """ Function to create a classifier. Valid types must contain the substrings
        'default' (for default with no algorithm),
        'GMM' (for Gaussian Mixture Model), or
        'SVM' (for Support Vector Machine), or
        'CART' or 'Tree' (for Classifier Tree);
        case insensitive.
    
    @param classifier_type String specifying a valid classifier type
    
    @param cutoff Number for rejection cutoff; valid matches must have a
        score above this value. Disregarded for classifier tree.
        
    @retval Classifier object
    
    """
    lower_case_str = classifier_type.lower()
    
    if 'gmm' in lower_case_str:
        return GMMClassifier(cutoff)
    elif 'svm' in lower_case_str:
        return SVMClassifier(cutoff)
    elif ('cart' in lower_case_str) or ('tree' in lower_case_str):
        return CARTClassifier(0.5)
    elif 'default' in lower_case_str:
        return Classifier(cutoff)
    else:
        raise ValueError('invalid classifier type %s' % classifier_type)