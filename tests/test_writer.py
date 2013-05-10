"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests picolo.writer
"""

import nose.tools
from picolo.matcher import Matcher
from picolo.writer import Writer
import numpy as np

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
        self.writer.draw_neighbors(self.matcher.dist_neighbors,
                                   'tests/data/sample_dist_neighbors.pdf')
                                   
    def test_uc(self):
        self.writer.draw_unitcell_diagnostics('test',
                                              'tests/data/sample_uc_diagnostics.pdf')
                                              
    def test_fraction(self):
        output = self.writer.write_fractions_matched()
        nose.tools.assert_equal(len(output), 3+2*len(self.matcher.shapes))
        nose.tools.assert_equal(output[0], self.matcher.name)
        
    def test_features(self):
        w_features = self.writer.write_features('test')
        m_features = self.matcher.feature_matrix('test')
        all_true = np.all(w_features == m_features)
        nose.tools.assert_true(all_true)
        
        