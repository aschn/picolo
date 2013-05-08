"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests picolo.writer
"""

from picolo.matcher import Matcher
from picolo.writer import Writer

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