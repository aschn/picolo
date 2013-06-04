"""
@package test_trainer
@author Anna Schneider
@version 0.1
@brief Tests picolo.trainer
"""

import nose.tools
import numpy as np
import logging
import math

from picolo.trainer import trainer_factory
from picolo.matcher import Matcher

class TestTrainer:
    
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        self.features = matcher_uc.feature_matrix('test')
        self.template_shape = matcher_uc.shapes['test']
        self.trainer_default = trainer_factory()
        self.n = 206

    def test_init_default(self):
        nose.tools.assert_equal(self.trainer_default._X.size, 0)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.algorithm, 'none')
        nose.tools.assert_equal(self.trainer_default.n_sources, 0)
        nose.tools.assert_equal(self.trainer_default.n_features, 0)
        nose.tools.assert_equal(self.trainer_default.n_classes, None)
                
    def test_load_unsupervised_first(self):
        self.trainer_default.load(self.features)
        nose.tools.assert_equal(self.trainer_default._X.size,
                                self.features.size)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.n_sources, 1)
        nose.tools.assert_equal(self.trainer_default.n_features,
                                self.features.shape[1]) 
        
    def test_load_unsupervised_later(self):
        self.trainer_default.load(self.features)
        self.trainer_default.load(self.features)
        nose.tools.assert_equal(self.trainer_default._X.size,
                                self.features.size*2)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.n_sources, 2)
        nose.tools.assert_equal(self.trainer_default.n_features,
                                self.features.shape[1]) 
        
    def test_load_supervised_first(self):
        self.trainer_default.load(self.features, np.ones(self.n))
        nose.tools.assert_equal(self.trainer_default._y.size, self.n)
        
    def test_load_supervised_later(self):
        self.trainer_default.load(self.features, np.ones(self.n))
        self.trainer_default.load(self.features, np.ones(self.n))
        nose.tools.assert_equal(self.trainer_default._y.size, self.n*2)
        
    @nose.tools.raises(ValueError)
    def test_load_yrows_error(self):
        self.trainer_default.load(self.features, np.ones(self.n/2))

    @nose.tools.raises(ValueError)
    def test_load_ycols_error(self):
        self.trainer_default.load(self.features, np.ones([self.n,2]))

    @nose.tools.raises(ValueError)
    def test_load_xcols_error(self):
        self.trainer_default.load(self.features)
        self.trainer_default.load(self.features[:,0])
        
    @nose.tools.raises(RuntimeError)
    def test_predict_default(self):
        self.trainer_default.predict()
        
    @nose.tools.raises(RuntimeError)
    def test_accuracy_default(self):
        self.trainer_default.accuracy()
        
class TestGMMTrainer:
                                                                        
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        self.features = matcher_uc.feature_matrix('test')
        self.template_shape = matcher_uc.shapes['test']
        self.trainer_gmm = trainer_factory('gmm')
        self.n = 206
        
    def test_init_gmm(self):
        nose.tools.assert_equal(self.trainer_gmm.algorithm, 'gmm')        
        nose.tools.assert_equal(self.trainer_gmm.n_classes, None)
        
    def test_fit_gmm_one(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        shapes = self.trainer_gmm.params_as_shapes(self.template_shape)
        nose.tools.assert_equal(len(shapes), 1)
        nose.tools.assert_almost_equal(shapes[0].get('a'),
                                       np.mean(self.features[:,0]))
        nose.tools.assert_almost_equal(shapes[0].get('b'),
                                       np.mean(self.features[:,1]))
        nose.tools.assert_almost_equal(shapes[0].get('degrees'),
                                       np.mean(self.features[:,2]))
        nose.tools.assert_almost_equal(shapes[0].get('sds')[0],
                                       np.std(self.features[:,0]),
                                       places=4)
        nose.tools.assert_almost_equal(shapes[0].get('sds')[1],
                                       np.std(self.features[:,1]),
                                       places=4)
        nose.tools.assert_almost_equal(shapes[0].get('sds')[2],
                                       np.std(self.features[:,2]),
                                       places=4)
        
    def test_means_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        means = self.trainer_gmm.means()
        nose.tools.assert_almost_equal(means[0,0],
                                       np.mean(self.features[:,0]))
        nose.tools.assert_almost_equal(means[0,1],
                                       np.mean(self.features[:,1]))
        nose.tools.assert_almost_equal(means[0,2],
                                       np.mean(self.features[:,2]))

    def test_sds_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        sds = self.trainer_gmm.sds()
        nose.tools.assert_almost_equal(sds[0,0],
                                       np.std(self.features[:,0]),
                                       places=4)
        nose.tools.assert_almost_equal(sds[0,1],
                                       np.std(self.features[:,1]),
                                       places=4)
        nose.tools.assert_almost_equal(sds[0,2],
                                       np.std(self.features[:,2]),
                                       places=4)

    def test_fit_gmm_mult(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        shapes = self.trainer_gmm.params_as_shapes(self.template_shape)
        nose.tools.assert_equal(len(shapes), 2)
            
    def test_aic_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        nose.tools.assert_almost_equal(self.trainer_gmm.aic(),
                                       5178.4171121803702)

    def test_bic_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        expected = self.trainer_gmm.aic() + 3*math.log(self.n) + 2*(3-1)
        nose.tools.assert_almost_equal(self.trainer_gmm.bic(), expected,
                                       places=1)
                         
    def test_predict_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit(n_classes=2)
        pred_labels = self.trainer_gmm.predict()
        nose.tools.assert_equal(pred_labels.size, self.n)
        nose.tools.assert_equal(len(np.unique(pred_labels)),
                                self.trainer_gmm.n_classes)
        nose.tools.assert_equal(len(np.unique(pred_labels)), 2)
    
    def test_accuracy_gmm(self):
        self.trainer_gmm.load(self.features, np.ones(self.n))
        self.trainer_gmm.fit(n_classes=2)
        accuracy = self.trainer_gmm.accuracy()
        pred_labels = self.trainer_gmm.predict()
        expected = np.count_nonzero(pred_labels) / float(pred_labels.size)
        nose.tools.assert_almost_equal(accuracy, expected)

class TestSVMTrainer:
                                                                        
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        self.features = matcher_uc.feature_matrix('test')
        self.template_shape = matcher_uc.shapes['test']
        self.trainer_svm = trainer_factory('svm')
        self.n = 206
        self.ys = np.array([matcher_uc.get_features('test',i).get('is_valid') for i in range(self.n)],
                            dtype=np.int)
        self.notys = np.array([not matcher_uc.get_features('test',i).get('is_valid') for i in range(self.n)],
                               dtype=np.int)
        
    def test_init_gmm(self):
        nose.tools.assert_equal(self.trainer_svm.algorithm, 'svm')  
        nose.tools.assert_equal(self.trainer_svm.n_classes, None)

    def test_fit_svm_1(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.fit()
        shapes = self.trainer_svm.params_as_shapes(self.template_shape)
        nose.tools.assert_equal(len(shapes), 1)
        nose.tools.assert_almost_equal(shapes[0].get('a'), 0.0088578582091631101)
        nose.tools.assert_almost_equal(shapes[0].get('b'), 0.011840954483356043)
        nose.tools.assert_almost_equal(shapes[0].get('degrees'), 0.029408398826550369)
        nose.tools.assert_almost_equal(shapes[0].get('intercept'), -0.997138, places=5)        

    def test_fit_svm_2(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.load(self.features, self.notys)
        self.trainer_svm.fit()
        nose.tools.assert_equal(self.trainer_svm.n_classes, 1)

    def test_fit_svm_3(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.load(self.features, self.notys+1)
        self.trainer_svm.fit()
        nose.tools.assert_equal(self.trainer_svm.n_classes, 3)
            
    def test_predict_svm(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.fit()
        pred_labels = self.trainer_svm.predict()
        nose.tools.assert_equal(pred_labels.size, self.n)
        nose.tools.assert_equal(len(np.unique(pred_labels)), 2)
        
    @nose.tools.raises(RuntimeWarning)
    def test_fit_svm_nclasses(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.fit(n_classes=5)

    @nose.tools.raises(ValueError)
    def test_fit_svm_noy(self):
        self.trainer_svm.load(self.features)
        self.trainer_svm.fit()
        
    def test_accuracy_svm(self):
        self.trainer_svm.load(self.features, self.ys)
        self.trainer_svm.fit()
        accuracy = self.trainer_svm.accuracy()
        nose.tools.assert_almost_equal(accuracy, 1)
