"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests picolo.matcher
"""

import nose.tools
import numpy as np

from picolo.matcher import Matcher

class TestMatcher:
    
    def setup(self):
        self.matcher = Matcher('tests/data/sample_config.xy')
        self.matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        
    def test_init(self):
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
        nose.tools.assert_greater(len(self.matcher.mask.mask), 0)
        
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
        
    def test_best_match(self):
        self.matcher.initialize_shapes('tests/data/sample_db_in.xml')
        names, vals = self.matcher.get_best_match('test')
        nose.tools.assert_equal(len(names), self.matcher.config.N)
        nose.tools.assert_equal(len(vals), self.matcher.config.N)
        for n in names:
            nose.tools.assert_equal(n, '')
    
    def test_classify(self):
        classes = self.matcher.classify()
        nose.tools.assert_equal(len(classes), self.matcher.config.N)
        for c in classes:
            nose.tools.assert_equal(c, 0)
            
    def test_count_match(self):
        classes = self.matcher.classify()
        counts = self.matcher.count_matched(classes)
        for ishape, count in enumerate(counts):
            nose.tools.assert_equal(count,
                                    np.count_nonzero(classes==(ishape+1)))
            
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
        areas = self.matcher.areas_matched(classes)
        for ishape, sname in enumerate(self.matcher_uc.shapes.names()):
            class_ids = self.matcher_uc.particles_matched(classes, sname)
            if len(class_ids) > 0:
                nose.tools.assert_greater(areas[ishape], 0)                                
                                