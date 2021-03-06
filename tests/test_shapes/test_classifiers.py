"""
@package test_classifiers
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Classifier
"""

import nose.tools
from scipy import stats

from picolo.shapes.classifiers import Classifier, SVMClassifier, GMMClassifier, CARTClassifier
from picolo.shapes.classifiers import classifier_factory
from picolo.shapes.shapes import Shape

class TestBaseClassifier:

    def setup(self):
        self.clf_default = Classifier()
        cutoff = -0.5
        self.clf_cut = Classifier(cutoff=cutoff)
       
        self.shape = Shape()

    def test_compute_match(self):
        match = self.clf_default.compute_match(self.shape, self.shape)
        nose.tools.assert_equal(match, 0)
        
    def test_is_match(self):
        default_val = self.clf_default.compute_match(self.shape, self.shape)
        cut_val = self.clf_cut.compute_match(self.shape, self.shape)
        nose.tools.assert_false(self.clf_default.is_match(self.shape, default_val))        
        nose.tools.assert_true(self.clf_cut.is_match(self.shape, cut_val))
        
    def test_algorithm(self):
        nose.tools.assert_equal(self.clf_default.algorithm(), 'default')
        
    def test_factory(self):
        clf_factory = classifier_factory()
        nose.tools.assert_false(clf_factory.is_match(None, 0))
        
class TestRealClassifiers:
    
    def setup(self):
        self.svm_class_shape = Shape(['a', 'b'], [1, 1], intercept=1)
        self.gmm_class_shape = Shape(['a', 'b'], [1, 1], sds=[1.1, 1.1])
        self.cart_class_shape = Shape([], [],
                                      rules=[('is_valid', True, 'eq'),
                                             ('a', 2, 'gt'),
                                             ('a', 2, 'ge'),
                                             ('b', 2, 'lt'),
                                             ('b', 2, 'le'),
                                             ('b', 2, 'ne')
                                             ])
        self.data_shape = Shape(['a', 'b'], [2.1, 0.1])        
    
    def test_svm_factory(self):
        svm_fact = classifier_factory('svm', 0.1)
        svm_built = SVMClassifier(0.1)
        nose.tools.assert_almost_equal(svm_fact.compute_match(self.svm_class_shape,
                                                              self.data_shape),
                                       svm_built.compute_match(self.svm_class_shape,
                                                              self.data_shape)
                                       )
                                       
    def test_svm_match_true(self):
        svm = classifier_factory('svm', 0.1)
        matchval = svm.compute_match(self.svm_class_shape, self.data_shape)
        expected = (1.0 * 2.1 + 1.0 * 0.1) + 1.0
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_true(svm.is_match(self.svm_class_shape, matchval))   

    def test_svm_match_false(self):
        svm = classifier_factory('svm', 0.1)
        matchval = svm.compute_match(self.svm_class_shape,
                                     Shape(['a', 'b'], [2.1, -3.1]))
        expected = (1.0 * 2.1 + 1.0 * -3.1) + 1.0
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_false(svm.is_match(self.svm_class_shape, matchval))   
        
    def test_svm_algorithm(self):
        svm = classifier_factory('svm', 0.1)
        nose.tools.assert_equal(svm.algorithm(), 'SVM')

    def test_gmm_factory(self):
        gmm_fact = classifier_factory('gmm', 0.1)
        gmm_built = GMMClassifier(0.1)
        nose.tools.assert_almost_equal(gmm_fact.compute_match(self.gmm_class_shape,
                                                              self.data_shape),
                                       gmm_built.compute_match(self.gmm_class_shape,
                                                              self.data_shape)
                                       )
    def test_gmm_match_true(self):
        gmm = classifier_factory('gmm', 0.1)
        matchval = gmm.compute_match(self.gmm_class_shape, self.gmm_class_shape)
        expected = stats.norm.pdf(1, loc=1, scale=1.1) * stats.norm.pdf(1, loc=1, scale=1.1)
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_true(gmm.is_match(self.gmm_class_shape, matchval))

    def test_gmm_match_false(self):
        gmm = classifier_factory('gmm', 0.1)
        matchval = gmm.compute_match(self.gmm_class_shape, self.data_shape)
        expected = stats.norm.pdf(2.1, loc=1, scale=1.1) * stats.norm.pdf(0.1, loc=1, scale=1.1)
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_false(gmm.is_match(self.gmm_class_shape, matchval))
        
    def test_gmm_algorithm(self):
        gmm = classifier_factory('gmm', 0.1)
        nose.tools.assert_equal(gmm.algorithm(), 'GMM')

    def test_cart_factory(self):
        cart_fact = classifier_factory('cart')
        cart_built = CARTClassifier(0.5)
        nose.tools.assert_almost_equal(cart_fact.compute_match(self.cart_class_shape,
                                                              self.data_shape),
                                       cart_built.compute_match(self.cart_class_shape,
                                                              self.data_shape)
                                       )      

    def test_cart_match_true(self):
        cart = classifier_factory('cart')
        matchval = cart.compute_match(self.cart_class_shape, self.data_shape)
        expected = 1
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_true(cart.is_match(self.cart_class_shape, matchval))

    def test_cart_match_false(self):
        cart = classifier_factory('cart')
        self.data_shape.invalidate()
        matchval = cart.compute_match(self.cart_class_shape, self.data_shape)
        expected = 0
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_false(cart.is_match(self.cart_class_shape, matchval))
             
    def test_cart_algorithm(self):
        cart = classifier_factory('cart')
        nose.tools.assert_equal(cart.algorithm(), 'CART')
        