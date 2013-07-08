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
        features_w_null = matcher_uc.feature_matrix('test')
        nonnull_inds = np.where(features_w_null.mean(axis=1))[0]
        self.features = features_w_null[nonnull_inds]
        self.template_shape = matcher_uc.shapes['test']
        self.trainer_default = trainer_factory()
        self.n = self.features.shape[0]

    def test_init_default(self):
        nose.tools.assert_equal(self.trainer_default._X.size, 0)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.algorithm, 'none')
        nose.tools.assert_equal(self.trainer_default.n_sources, 0)
        nose.tools.assert_equal(self.trainer_default.n_points, 0)
        nose.tools.assert_equal(self.trainer_default.n_features, 0)
        nose.tools.assert_equal(self.trainer_default.n_classes, None)
        
    @nose.tools.raises(ValueError)        
    def test_load_invalid(self):
        self.features[0,1] = np.nan
        self.trainer_default.load(self.features)

    def test_load_unsupervised_first(self):
        self.trainer_default.load(self.features)
        nose.tools.assert_equal(self.trainer_default._X.size,
                                self.features.size)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.n_sources, 1)
        nose.tools.assert_equal(self.trainer_default.n_points,
                                self.features.shape[0]) 
        nose.tools.assert_equal(self.trainer_default.n_features,
                                self.features.shape[1]) 
        
    def test_load_unsupervised_later(self):
        self.trainer_default.load(self.features)
        self.trainer_default.load(self.features)
        nose.tools.assert_equal(self.trainer_default._X.size,
                                self.features.size*2)
        nose.tools.assert_equal(self.trainer_default._y.size, 0)
        nose.tools.assert_equal(self.trainer_default.n_sources, 2)
        nose.tools.assert_equal(self.trainer_default.n_points,
                                self.features.shape[0]*2) 
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
        
    def test_clear(self):
        self.trainer_default.load(self.features, np.ones(self.n))
        self.trainer_default.clear()
        nose.tools.assert_equal(self.trainer_default.n_sources, 0)
        nose.tools.assert_equal(self.trainer_default.n_features, 0)
        nose.tools.assert_equal(self.trainer_default.n_points, 0)
        
    @nose.tools.raises(RuntimeError)
    def test_predict_default(self):
        self.trainer_default.predict()
        
    @nose.tools.raises(RuntimeError)
    def test_accuracy_default(self):
        self.trainer_default.accuracy()
        
    @nose.tools.raises(RuntimeError)
    def test_cv_default(self):
        self.trainer_default.cv_error(k=3)
        
    def test_pairs(self):
        self.trainer_default.load(self.features)
        for iplot, ix, xarr, iy, yarr in self.trainer_default.pairs():
            iy_expected = (iplot-1) / self.trainer_default.n_features 
            nose.tools.assert_equal(iy_expected, iy)
            nose.tools.assert_true(np.all(self.features[:,iy] == yarr))
            ix_expected = (iplot-1) % self.trainer_default.n_features
            nose.tools.assert_equal(ix_expected, ix)
            nose.tools.assert_true(np.all(self.features[:,ix] == xarr))
        
        
class TestGMMTrainer:
                                                                        
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        features_w_null = matcher_uc.feature_matrix('test')
        nonnull_inds = np.where(features_w_null.mean(axis=1))[0]
        self.features = features_w_null[nonnull_inds]
        self.template_shape = matcher_uc.shapes['test']
        self.trainer_gmm = trainer_factory('gmm')
        self.n = self.features.shape[0]
        
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
                                       places=3)
        nose.tools.assert_almost_equal(shapes[0].get('sds')[1],
                                       np.std(self.features[:,1]),
                                       places=3)
        nose.tools.assert_almost_equal(shapes[0].get('sds')[2],
                                       np.std(self.features[:,2]),
                                       places=3)
        
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
                                       places=3)
        nose.tools.assert_almost_equal(sds[0,1],
                                       np.std(self.features[:,1]),
                                       places=3)
        nose.tools.assert_almost_equal(sds[0,2],
                                       np.std(self.features[:,2]),
                                       places=3)

    def test_fit_gmm_mult(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        shapes = self.trainer_gmm.params_as_shapes(self.template_shape)
        nose.tools.assert_equal(len(shapes), 2)
        
    def test_get_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit(n_classes=3)
        params = self.trainer_gmm.get_params()
        means = self.trainer_gmm.means()
        sds = self.trainer_gmm.sds()
        nose.tools.assert_true(np.all(params[:,:,0] == means))
        nose.tools.assert_true(np.all(params[:,:,1] == sds))
        
    def test_set_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit(n_classes=2)

        means = np.arange(6).reshape((2,3))
        sds = np.arange(6).reshape((2,3)) * 0.1
        params = np.dstack((means, sds))
        
        self.trainer_gmm.set_params(means=means)
        print means
        print self.trainer_gmm.means()
        nose.tools.assert_true(np.all(means == self.trainer_gmm.means()))

        self.trainer_gmm.set_params(sds=sds)
        nose.tools.assert_true(np.all(sds == self.trainer_gmm.sds()))

        self.trainer_gmm.set_params(params=params)
        nose.tools.assert_true(np.all(params == self.trainer_gmm.get_params()))

    def test_aic_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        expected = 2.0 * self.trainer_gmm.n_classes * self.trainer_gmm.n_features * 2
        expected -= 2.0 * self.trainer_gmm._classifier.score(self.trainer_gmm._X).sum()
        nose.tools.assert_almost_equal(self.trainer_gmm.aic(), expected)

    def test_bic_gmm(self):
        # TO DO
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit()
        expected = self.trainer_gmm.aic() 
        expected += (math.log(self.n) - 2.0) * self.trainer_gmm.n_classes * self.trainer_gmm.n_features * 2.0
        nose.tools.assert_almost_equal(self.trainer_gmm.bic(), expected)
                         
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

    def test_bootstrap_fit_gmm(self):
        self.trainer_gmm.load(self.features)
        self.trainer_gmm.fit(n_classes=2)
        means_expect = self.trainer_gmm.means()
        sds_expect = self.trainer_gmm.sds()
        n_reps = 100
        means_boot = np.zeros([n_reps, 2, self.trainer_gmm.n_features])
        sds_boot = np.zeros([n_reps, 2, self.trainer_gmm.n_features])
        bootstrap_generator = self.trainer_gmm.bootstrap_fit(n_reps=n_reps, n_classes=2,
                                              labels_true=self.trainer_gmm.predict(),
                                              seed=1)
        for irep, (model, inds) in enumerate(bootstrap_generator):
            means_boot[irep] = model.means()
            sds_boot[irep] = model.sds()
        means_est = np.mean(means_boot, axis=0)
        sds_est = np.mean(sds_boot, axis=0)
        means_CI = np.std(means_boot, axis=0, ddof=1) * 1.96
        sds_CI = np.std(sds_boot, axis=0, ddof=1) * 1.96
        nose.tools.assert_true(np.all((means_CI - np.abs(means_est - means_expect)) / means_CI > 0.5))
        nose.tools.assert_true(np.all((sds_CI - np.abs(sds_est - sds_expect)) / sds_CI > 0.5))

    def test_bootstrap_select_gmm(self):
        self.trainer_gmm.load(self.features)
        n_reps = 100
        ks = range(1,4)
        bootstrap_generator = self.trainer_gmm.bootstrap_select(n_reps, ks)
        count = 0
        for best_k, bics in bootstrap_generator:
            best_ik_k = ks.index(best_k)
            best_ik_bic = np.argmin(bics)
            nose.tools.assert_equal(best_ik_k, best_ik_bic)
            count += 1
        nose.tools.assert_equal(count, n_reps)
        
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
        nose.tools.assert_almost_equal(shapes[0].get('a'), 0.0097909, places=5)
        nose.tools.assert_almost_equal(shapes[0].get('b'), 0.0109956, places=5)
        nose.tools.assert_almost_equal(shapes[0].get('degrees'), 0.032635, places=5)
        nose.tools.assert_almost_equal(shapes[0].get('intercept'), -0.997127, places=3)        

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

    def test_cv_svm(self):
        self.trainer_svm.load(self.features, self.ys)
        cve = self.trainer_svm.cv_error(k=3)
        nose.tools.assert_almost_equal(cve, 0)