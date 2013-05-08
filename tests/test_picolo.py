"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests picolo.picolo
"""

import nose.tools

from picolo import Matcher, Writer

class TestMatcher:
    
    def setup(self):
        self.matcher = Matcher('tests/data/sample_config.xy')
        
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
            
    def test_connected_components(self):
        props = [True for i in range(10)] + [False for i in range(self.matcher.config.N-10)]
        clusters, uncluster = self.matcher.connected_components(props, ntype='delaunay')
        nose.tools.assert_equal(len(uncluster), self.matcher.config.N-10)
        nose.tools.assert_greater(len(clusters), 1)
        for i in range(10):
            nose.tools.assert_true(i in set.union(*clusters))
            
    def test_fraction_matched(self):
        nose.tools.assert_true(True)
        
class TestWriter:
     
    def setup(self):
        self.matcher = Matcher('tests/data/sample_config.xy',
                               name='tester',
                               xmlname='tests/data/sample_db_uc.xml',
                               imname='tests/data/sample_mask.tif',
                               lx=500, ly=500)
        self.writer = Writer(self.matcher)
        
    def test_classification(self):
        self.writer.draw_classification('tests/data/sample_classification.pdf')
 
    def test_neighbors(self):
        self.writer.draw_neighbors(self.matcher.delaunay_neighbors,
                                   'tests/data/sample_neighbors.pdf')
                                   
    def test_uc(self):
        self.writer.draw_unitcell_diagnostics('test',
                                              'tests/data/sample_uc_diagnostics.pdf')