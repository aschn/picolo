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
                                              
    def test_fraction_particles(self):
        output = self.writer.write_fraction_particles_matched()
        nose.tools.assert_equal(len(output), 2+len(self.matcher.shapes))
        nose.tools.assert_equal(output[0], self.matcher.name)
        
    def test_fraction_area(self):
        output = self.writer.write_fraction_area_matched()
        nose.tools.assert_equal(len(output), 2+len(self.matcher.shapes))
        nose.tools.assert_equal(output[0], self.matcher.name)
        
    def test_features(self):
        w_features = self.writer.write_features('test')
        m_features = self.matcher.feature_matrix('test')
        all_true = np.all(w_features == m_features)
        nose.tools.assert_true(all_true)
        
    def test_radial(self):
        w_radial = self.writer.write_radial_distribution()
        m_radial = self.matcher.config.radial_distribution()
        all_true = np.all(w_radial == m_radial)
        nose.tools.assert_true(all_true)

    def test_nnd(self):
        w_nnd = self.writer.write_nearest_neighbor_distribution()
        m_nnd = self.matcher.config.nearest_neighbor_distribution()
        all_true = np.all(w_nnd == m_nnd)
        nose.tools.assert_true(all_true)
       