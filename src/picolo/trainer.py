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
import itertools

# import external packages
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics

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
        
    @property
    def is_supervised(self):
        """ True if expected y values have been loaded, False if not. """
        return self._y.size > 0
        
    def _filter_X(self, data_id):
        if data_id is None:
            return self._X   
        else:
            try:
                if len(data_id) > 0:
                    return self._X[data_id,:]
                else:
                    raise ValueError("Can't filter X on data ids %s" % data_id)
            except TypeError:
                return self._X[self._data_sources==data_id,:]
            
    def _filter_y(self, data_id):
        if data_id is None:
            return self._y   
        else:
            try:
                if len(data_id) > 0:
                    return self._y[data_id]
                else:
                    raise ValueError("Can't filter y on data ids %s" % data_id)
            except TypeError:
                return self._y[self._data_sources==data_id]
            
    def load(self, x_vals, y_vals=None):
        """ Load data into design matrix, with or without observations.
            Appends to existing data.
            
        @param x_vals NxM ndarray of feature data
        
        @param y_vals Nx1 ndarray of observation data, optional
            
        """
        # check for nans and infs
        isfinite = np.isfinite(x_vals)
        if not np.all(isfinite):
            rows_w_error = np.where(np.logical_not(np.all(isfinite, axis=1)))
            raise ValueError("Invalid values found in rows %s" % rows_w_error)
        
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
        ids = np.ones(x_vals.shape[0]) * self.n_sources
        self._data_sources = np.hstack((self._data_sources, ids))
    
    def get_params(self):
        """ Get model-specific parameters """
        pass
    
    def set_params(self, params=None):
        pass
    
    def fit(self, data_id=None, n_classes=None):
        """ Fit the model using the loaded data. 
        
        @param data_id portion of loaded data to use
        
        @param n_classes Number of classes in fitted model
        
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
            return self._classifier.predict(self._filter_X(data_id))                
    
    def aic(self, data_id=None):
        """ Returns the Akaike Information Criterion of the fitted model.  
        
        @param data_id portion of loaded data to use
        
        @retval aic Number for AIC
        
        """
        if self._classifier is None:
            raise RuntimeError("Must fit Trainer before getting AIC.")
        else:
            return self._classifier.aic(self._filter_X(data_id))

    def bic(self, data_id=None):
        """ Returns the Bayes Information Criterion of the fitted model.  
        
        @param data_id portion of loaded data to use
        
        @retval bic Number for BIC

        """
        if self._classifier is None:
            raise RuntimeError("Must fit Trainer before getting BIC.")
        else:
            return self._classifier.bic(self._filter_X(data_id))
        
    def accuracy(self, data_id=None):
        """ Get fraction of correct predictions of fitted (supervised) model.
        
        @param data_id portion of loaded data to use
        
        @retval fraction_correct as float
        
        """
        ydata = self._filter_y(data_id)
            
        if ydata.size == 0:
            raise RuntimeError("Can't get accuracy without y vector.")
        else:
            n_success = np.count_nonzero(self.predict(data_id) == ydata)
            return float(n_success) / ydata.size
            
    
    def cv_error(self, k, n_classes=None):
        """ Get k-fold cross-validation error of a (supervised) model.
        
        @param k Number of cross-validation folds
        
        @param n_classes Number of classes in fitted model; 
            if None, use default
        
        @retval cv_err Average CV error as float
        
        """
        if (not self.is_supervised) or (self._data_sources.size < k):
            raise RuntimeError("Can't get CV error without enough loaded data.")
            
        # set up for CV
        skf = StratifiedKFold(self._y, k)
        cvs = np.zeros(k)
        
        # loop over folds
        for icv, (train, test) in enumerate(skf):
            # fit with training data
            self.fit(data_id=train, n_classes=n_classes)
            
            # supervised error is fraction of mistakes
            cvs[icv] = 1.0 - self.accuracy(data_id=test)
                
        # return
        return np.mean(cvs)
    
    def params_as_shapes(self, template_shape):
        """ Get params of fitted model(s) as Shape objects.
        
        @param template_shape Shape object with the same features as the
            design matrix
            
        @retval shapes List of Shapes, length n_classes
        
        """
        return []
    
    def pairs(self):
        """ Generator to yield data to make plot like R pairs.
        
        @retval iplot Number for plt.subplot(trainer.n_features, trainer.n_features, iplot)
        
        @retval ix Number for index of x variable
        
        @retval xarr Nx1 ndarray of numbers of x values
        
        @retval iy Number for index of y variable

        @retval yarr Nx1 ndarray of numbers of y values
        
        """
        iplot = 0
        for iy in range(self.n_features):
            for ix in range(self.n_features):
                iplot += 1
                if ix != iy:
                    yield iplot, ix, self._X[:,ix], iy, self._X[:,iy]
                    
    def confusion_matrix(self, labels_true):
        """ Return confusion matrix between fitted model and given labels.

        @param labels_true Nx1 ndarray of 'true' labels

        @retval conf_matrix [n_classes, n_classes] ndarray where
            conf_matrix[i,j] is number of points in class i of given labels
                and in class j of fitted model
        
        """
        return metrics.confusion_matrix(labels_true, self.predict())   
                    
    def bootstrap_fit(self, n_reps, n_classes=None, labels_true=None,
                  seed=None):
        """ Generate n_reps models by fitting to bootstrapped data.
            Classes are matched to the given labels, or to the predicted
                labels on the complete dataset if no labels are given.
        
        @param n_reps Number of times to bootstrap
        
        @param n_classes Number of classes in fitted model; 
            if None, use default
            
        @param labels_true Optional ndarray of class labels, length n_rows
            
        @param seed Optional number for seeding rng (only use for testing)            
            
        @retval models fitted with bootstrapped data

        @retval data_ids Ndarray of indices for selected rows
            
        """
        # set up storage
        if not n_classes:
            n_classes = self.n_classes
        if (not self._classifier) or (n_classes != self.n_classes):
            self.fit(n_classes=n_classes)
        if seed is not None:
            np.random.seed(seed)
        params_permut = np.zeros_like(self.get_params())
        
        # set up target labels
        if labels_true is not None: # check given labels
            if labels_true.size != self._X.shape[0]:
                raise ValueError("Got %d labels but expected %d." % (labels_true.size,
                                                                     self._X.shape[0]))
            if not np.all(np.unique(labels_true) == np.arange(n_classes)):
                raise ValueError("Labels have %d classes but expected %d." % (np.unique(labels_true).size,
                                                                              n_classes))
        else: # get labels from raw data
            self.fit(n_classes=n_classes)
            labels_true = self.predict()
            
        # do n reps of bootstrapping
        for irep in range(n_reps):
            # sample rows
            bootstrap_inds = np.random.randint(self._X.shape[0],
                                               size=self._X.shape[0])
            # get fitted labels
            self.fit(data_id=bootstrap_inds, n_classes=n_classes)
            labels_boot = self.predict()
            
            # find best permutation of labels to match expected labels
            best_permut = None
            best_error = np.Inf
            for permut in itertools.permutations(range(n_classes)):
                # permute labels
                labels_permut = [permut[l] for l in labels_boot]
                
                # get error
                is_misclassified = labels_permut != labels_true
                error = np.count_nonzero(is_misclassified)
                
                # save if better than previous error
                if error < best_error:
                    best_error = error
                    best_permut = permut
                if best_error == 0:
                    break
                
            # store params using permuted labels
            params = self.get_params()
            for iclass in range(n_classes):
                params_permut[best_permut[iclass]] = params[iclass]
            self.set_params(params=params_permut)
            
            # return
            yield self, bootstrap_inds

    def bootstrap_select(self, n_reps, ks, seed=None):
        """ Generate n_reps models by fitting to bootstrapped data
            and selecting the best model out of n_classes using BIC.
        
        @param n_reps Number of times to bootstrap
        
        @param ks Iterable of numbers of classes to select from
                        
        @param seed Optional number for seeding rng (only use for testing)            
            
        @retval best_ks Numbers of classes in the selected models

        @retval bics Numbers for BICs of all models
            
        """
        # set up storage
        if len(ks) == 0:
            raise ValueError("n_classes must contain at least 1 value of k.")
        if seed is not None:
            np.random.seed(seed)
            
        # do n reps of bootstrapping
        for irep in range(n_reps):
            # sample rows
            bootstrap_inds = np.random.randint(self._X.shape[0],
                                               size=self._X.shape[0])
            # get BIC for each k
            bics = np.zeros(len(ks))
            for ik, k in enumerate(ks):
                self.fit(data_id=bootstrap_inds, n_classes=k)
                bics[ik] = self.bic(bootstrap_inds)
                
            # best BIC is min
            best_ik = np.argmin(bics)
            best_k = ks[best_ik]

            # return
            yield best_k, bics
            
    
class GMMTrainer(Trainer):
    """ Trainer for a Gaussian mixture model.
        This is a wrapper around sklearn.mixture.GMM.
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
        self._classifier.fit(self._filter_X(data_id))
        
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
 
    def get_params(self):
        """ Get means and standard deviations of Gaussians.
        
        @retval params ndarray of shape [n_classes, n_features, 2]
            where params[:,:,0] is means and params[:,:,1] is SDs
        
        """
        return np.dstack((self.means(), self.sds()))
    
    def set_params(self, means=None, sds=None, params=None):
        """ Set means and standard deviations of Gaussians.
            If params is specified, overrides means and sds arguments.
        
        @param means [n_classes, n_features] ndarray of means (mus)
        
        @param sds [n_classes, n_features] ndarray of standard deviations
        
        @param params [n_classes, n_features, 2] ndarray of means and sds
        
        """
        if params is not None:
            means = params[:,:,0]
            sds = params[:,:,1]
            
        if means is not None:
            self._classifier.means_ = means
        if sds is not None:
            self._classifier.covars_ = np.square(sds)
    
                  
class SVMTrainer(Trainer):
    """ Trainer for a Support Vector Machine.
        This is a wrapper around sklearn.svm.LinearSVC
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
            self._classifier.fit(self._filter_X(data_id),
                                 self._filter_y(data_id))
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