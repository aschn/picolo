"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests picolo.matcher
"""

import nose.tools
import numpy as np
import math
import logging

from picolo.matcher import Matcher

class TestMatcherNull:
    
    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        self.matcher = Matcher('tests/data/sample_config.xy')
        
    def test_init_default(self):
        nose.tools.assert_equal(self.matcher.name, '')
        nose.tools.assert_equal(self.matcher.training_ids, [])
        nose.tools.assert_equal(len(self.matcher.shapes), 0)
        nose.tools.assert_equal(self.matcher.mask, None)
        nose.tools.assert_greater(self.matcher.config.N, 0)
        nose.tools.assert_greater(len(self.matcher.delaunay_neighbors), 0)
        nose.tools.assert_equal(len(self.matcher.dist_neighbors), 0)
        
    def test_remove_mask(self):
        self.matcher.remove_mask()
        nose.tools.assert_equal(self.matcher.mask, None)

    def test_initialize_mask(self):
        self.matcher.initialize_mask('tests/data/sample_mask.tif', 500, 500)
        nose.tools.assert_greater(self.matcher.mask.area(), 0)
        
    def test_initialize_shapes(self):
        self.matcher.initialize_shapes('tests/data/sample_db_in.xml')
        nose.tools.assert_greater(len(self.matcher.shapes), 0)
        
    def test_set_features(self):
        self.matcher.initialize_shapes('tests/data/sample_db_in.xml')
        self.matcher.set_features()
        nose.tools.assert_greater(len(self.matcher.get_features('test', 0)), 0)
        
    def test_raw_match(self):
        self.matcher.initialize_shapes('tests/data/sample_db_in.xml')
        vals = self.matcher.get_raw_match('test')
        nose.tools.assert_equal(len(vals), self.matcher.config.N)
        for v in vals:
            nose.tools.assert_almost_equal(v, 0)
        
    def test_best_match_null(self):
        self.matcher.initialize_shapes('tests/data/sample_db_in.xml')
        names, vals = self.matcher.get_best_match('test')
        nose.tools.assert_equal(len(names), self.matcher.config.N)
        nose.tools.assert_equal(len(vals), self.matcher.config.N)
        for n in names:
            nose.tools.assert_equal(n, '')

    def test_classify_null(self):
        classes = self.matcher.classify()
        nose.tools.assert_equal(len(classes), self.matcher.config.N)
        for c in classes:
            nose.tools.assert_equal(c, 0)                    
                  
class TestMatcherReal:
    
    def setup(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s | %(levelname)s | %(funcName)s |%(message)s')
        
        self.matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        self.n = 206
        self.n_valid = 72
        self.n_matched = 41
        
    def test_init_uc(self):
        nose.tools.assert_equal(self.matcher_uc.name, 'tester')
        nose.tools.assert_equal(self.matcher_uc.training_ids, [])
        nose.tools.assert_equal(len(self.matcher_uc.shapes), 1)
        nose.tools.assert_almost_equal(self.matcher_uc.mask.area(), 107067)
        nose.tools.assert_equal(self.matcher_uc.config.N, self.n)
        nose.tools.assert_greater(len(self.matcher_uc.delaunay_neighbors), 0)
        nose.tools.assert_equal(len(self.matcher_uc.dist_neighbors), self.n)
        nose.tools.assert_almost_equal(self.matcher_uc.dist_neighbors._r, 30)
    
    def test_get_features(self):
        shape = self.matcher_uc.get_features('test', 98)
        nose.tools.assert_almost_equal(shape.get('a'),
                                       20.7394123062)
        nose.tools.assert_almost_equal(shape.get('b'),
                                       23.0739478289)
        nose.tools.assert_almost_equal(shape.get('degrees'),
                                       72.1918282397)
        nose.tools.assert_almost_equal(shape.get('theta'),
                                       math.radians(72.1918282397))
               
    def test_best_match_uc(self):
        names, vals = self.matcher_uc.get_best_match('test')
        nose.tools.assert_equal(names.count('test'), self.n_matched)
        nose.tools.assert_equal(names.count(''), self.n-self.n_matched)
        nose.tools.assert_equal(vals.count(1), self.n_matched)        
        nose.tools.assert_equal(vals.count(0), self.n-self.n_matched) 
        
    def test_classify_uc(self):
        classes = self.matcher_uc.classify()
        nose.tools.assert_equal(np.count_nonzero(classes), self.n_matched)
        nose.tools.assert_equal(np.sum(classes), self.n_matched)
           
    def test_count_match(self):
        classes = self.matcher_uc.classify()
        counts = self.matcher_uc.count_matched(classes)
        for ishape, count in enumerate(counts):
            nose.tools.assert_equal(count,
                                    np.argwhere(np.asarray(classes)==(ishape+1)).size)
        nose.tools.assert_equal(counts, [self.n_matched])
            
    def test_particles_matched(self):
        classes = self.matcher_uc.classify()
        all_ids = self.matcher_uc.particles_matched(classes)
        nose.tools.assert_equal(len(all_ids), np.count_nonzero(classes))
        for iclass, sname in enumerate(self.matcher_uc.shapes.class_names()):
            if sname is not '':
                class_ids = self.matcher_uc.particles_matched(classes, sname)
                nose.tools.assert_equal(len(class_ids),
                                        np.count_nonzero(classes==iclass))    
                                
    def test_area_matched(self):
        classes = self.matcher_uc.classify()
        areas = self.matcher_uc.areas_matched(classes)
        for ishape, sname in enumerate(self.matcher_uc.shapes.names()):
            class_ids = self.matcher_uc.particles_matched(classes, sname)
            if len(class_ids) > 0:
                nose.tools.assert_greater(areas[ishape], 0)   

    def test_normalize_feature_matrix(self):
        norm_features = self.matcher_uc.feature_matrix('test',
                                                       normalize=True)
        expected_shape = (self.matcher_uc.config.N,
                          len(self.matcher_uc.shapes['test'].get_components()))
        nose.tools.assert_equal(norm_features.shape, expected_shape)  
        for ip in range(self.matcher_uc.config.N):
            if self.matcher_uc.get_features('test', ip).get('is_valid'):
                expected_norm = math.sqrt(sum([v*v for v in norm_features[ip]]))
                nose.tools.assert_almost_equals(expected_norm, 1) 
            else:   
                nose.tools.assert_almost_equals(sum(norm_features[ip]), 0)  
                                
    def test_feature_matrix(self):        
        features = self.matcher_uc.feature_matrix('test')
        count_valid = 0
        for ip in range(self.matcher_uc.config.N):
            shape = self.matcher_uc.get_features('test', ip)
            if shape.get('is_valid'):
                for iv, val in enumerate(shape.get_vals()):
                    nose.tools.assert_almost_equals(val, features[ip,iv]) 
                count_valid += 1                
        nose.tools.assert_equal(count_valid, self.n_valid)
  