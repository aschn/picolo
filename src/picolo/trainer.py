"""
@package picolo
@author Anna Schneider
@version 0.1
@brief Contains classes for Trainer, GMMTrainer, and SVMTrainer,
    and factory method trainer_factory
"""

# import from standard library
import logging
import copy

# import external packages
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM

class Trainer:
    """ Base class for training classifiers
            and converting into Shape objects.
        Interface is similar to scikit-learn classifiers.
        Trainer.fit, Trainer.n_classes, Trainer.algorithm,
            and Trainer.params_as_shapes
            should be overwritten in any derived class.
    """
    def __init__(self):
        """ Initialize trainer with empty X and y. """
        # design matrix
        self._X = np.array([], dtype=np.float)
        
        # target vector
        self._y = np.array([], dtype=np.int)
        
        # data sources
        self._data_sources = np.array([], dtype=np.int)
        
        # classifier
        self._classifier = None
        
    @property
    def n_sources(self):
        """ Number of loaded data sources. """
        return len(np.unique(self._data_sources))

    @property
    def n_classes(self):
        """ Number of predicted/fitted classes. """
        return None

    @property
    def n_features(self):
        """ Number of features, ie columns in the design matrix. """
        try:
            return self._X.shape[1]
        except IndexError:
            return 0
        
    @property
    def algorithm(self):
        """ Name of classifier algorithm used for training model """
        return 'none'
    
    def load(self, x_vals, y_vals=None):
        """ Load data into design matrix, with or without observations.
            Appends to existing data.
            
        @param x_vals NxM ndarray of feature data
        
        @param y_vals Nx1 ndarray of observation data, optional
            
        """
        # process observation vector
        if y_vals is not None:
            # test for vector
            if len(y_vals.shape) > 1:
                msg = "y_vals must be a vector. "
                msg += "Got y_vals cols %d" % (y_vals.shape[1])
                raise ValueError(msg)
            # test for matching number of rows
            if y_vals.shape[0] != x_vals.shape[0]:
                msg = "x_vals and y_vals must have same number of rows. "
                msg += "Got x_vals rows %d, y_vals rows %d" % (x_vals.shape[0],
                                                               y_vals.shape[0])
                raise ValueError(msg)
            # append if ok
            self._y = np.hstack((self._y, y_vals))
       
        # append design matrix     
        try:
            self._X = np.vstack((self._X, x_vals)) 
            logging.debug('appended to X')
        except (ValueError, IndexError):
            if self._X.size == 0:
                self._X = x_vals
                logging.debug('overwrote X')
            else:
                msg = "Couldn't append x_vals of shape %s " % str(x_vals.shape)
                msg += "to existing data with ncols %d." % self.n_features
                raise ValueError(msg)
                
        # append data sources
        ids = np.ones(x_vals.shape[1]) * self.n_sources
        self._data_sources = np.hstack((self._data_sources, ids))
    
    def fit(self, data_id=None, n_classes=None):
        """ Fit the model using the loaded data. 
        
        @param data_id portion of loaded data to use
        
        @param n_components Number of classes in fitted model
        
        @retval self
        
        """
        return self
    
    def predict(self, data_id=None):
        """ Get prediction from fitted model.
        
        @param data_id portion of loaded data to use
        
        @retval pred_labels Nx1 ndarray of predicted labels (ints)
        
        """
        if self._classifier is None:
            raise RuntimeError("Must fit Trainer before predicting.")
        else:
            return self._classifier.predict(self._X)
    
    def accuracy(self, data_id=None):
        """ Get fraction of correct predictions of fitted model.
        
        @param data_id portion of loaded data to use
        
        @retval fraction_correct as float
        
        """
        if self._y.size == 0:
            raise RuntimeError("Can't get accuracy without y vector.")
        else:
            n_success = np.count_nonzero(self.predict(data_id) == self._y)
            return float(n_success) / self._y.size
    
    def cv_error(self, k):
        """ TO DO.
            Get k-fold cross-validation error of fitted model.
        
        @param k Number of cross-validation folds.
        
        @retval cv_err as float
        
        """
        pass
    
    def params_as_shapes(self, template_shape):
        """ Get params of fitted model(s) as Shape objects.
        
        @param template_shape Shape object with the same features as the
            design matrix
            
        @retval shapes List of Shapes, length n_classes
        
        """
        pass

    
class GMMTrainer(Trainer):
    """ Trainer that uses scikit-learn to train a Gaussian mixture model.
        The number of fitted classes defaults to the number of data sources.
    """
    @property
    def algorithm(self):
        return 'gmm'

    @property
    def n_classes(self):
        if self._classifier:
            return self._classifier.n_components
        else:
            return None
        
    def fit(self, data_id=None, n_classes=None):
        # set n_classes
        if not n_classes:
            n_classes = self.n_sources
            
        # initialize GMM
        self._classifier = GMM(n_classes, covariance_type='diag')
        
        # fit GMM
        self._classifier.fit(self._X)
        
        # return
        return self
        
    def params_as_shapes(self, template_shape):
        # set up storage
        shapes = []
        
        # iterate over classes
        for iclass in range(self.n_classes):
            
            # copy template
            shape = copy.deepcopy(template_shape)
            
            # set components as means
            for ivar, var in enumerate(template_shape.get_components()):
                shape.put_component(var, self._classifier.means_[iclass, ivar])
                
            # set sd params
            sds = np.sqrt(self._classifier.covars_[iclass])
            shape.put_param('sds', sds)
            
            # append to storage
            shapes.append(shape)
            
        # return
        return shapes
        
    def means(self):
        """ Returns means of fitted model
            as (n_classes, n_features) ndarray. """
        return self._classifier.means_
        
    def sds(self):
        """ Returns standard deviations of fitted model
            as (n_classes, n_features) ndarray. """
        return np.sqrt(self._classifier.covars_)
 
    def aic(self):
        """ Returns the AIC of the fitted model. """
        if self._classifier is None:
            raise RuntimeError("Must fit Trainer before getting AIC.")
        else:
            return self._classifier.aic(self._X)

    def bic(self):
        """ Returns the BIC of the fitted model. """
        if self._classifier is None:
            raise RuntimeError("Must fit Trainer before getting BIC.")
        else:
            return self._classifier.bic(self._X)
        
           
class SVMTrainer(Trainer):
    """ Trainer that uses scikit-learn to train a Support Vector Machine.
        A linear kernel function is used for easy interpretability.
        Requires observations with at least 2 distinct class labels.
        Multiclass fitting is one-to-rest.
    """    
    @property
    def algorithm(self):
        return 'svm'
        
    @property
    def n_classes(self):
        if self._classifier:
            return self._classifier.intercept_.size
        else:
            return None
        
    def fit(self, data_id=None, n_classes=None):
        # initialize SVM
        self._classifier = LinearSVC(class_weight='auto')
        
        # fit SVM
        try:
            self._classifier.fit(self._X, self._y)
        except ValueError:
            raise ValueError("SVM needs y vector with at least 2 classes for fitting.")
            
        # warn if n_classes given
        if n_classes is not None:
            if n_classes != self._classifier.coef_.shape[0]:
                msg = "In SVM fit, asked for %d classes " % n_classes
                msg += " but got %d." % self._classifier.coef_.shape[0]
                raise RuntimeWarning(msg)
                
        # return
        return self
        
    def params_as_shapes(self, template_shape):
        # set up storage
        shapes = []
        
        # iterate over classes
        for iclass in range(self.n_classes):
            
            # copy template
            shape = copy.deepcopy(template_shape)
            
            # set components as coefs
            for ivar, var in enumerate(template_shape.get_components()):
                coef = self._classifier.coef_[iclass, ivar]
                shape.put_component(var, coef)
                
            # set intercept param
            shape.put_param('intercept', self._classifier.intercept_[iclass])
            
            # append to storage
            shapes.append(shape)
            
        # return
        return shapes
        

def trainer_factory(algorithm=None):
    """ Factory method to make a Trainer using a particular algorithm.
        Options are:
            'gmm': Gaussian Mixture Model,
            'svm': Support Vector Machine,
            None: useless base class.
    """
    if algorithm is None:
        return Trainer()
    elif algorithm == 'gmm':
        return GMMTrainer()
    elif algorithm == 'svm':
        return SVMTrainer()
    else:
        raise ValueError("Cannot build Trainer of type %s." % algorithm)