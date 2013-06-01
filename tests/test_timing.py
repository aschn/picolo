"""
@package test_picolo
@author Anna Schneider
@version 0.1
@brief Tests timing for picolo.matcher
"""

import nose.tools

from picolo.matcher import Matcher

class TestMatcherTiming:
    
    def setup(self):
        self.matcher_uc = Matcher('tests/data/sample_config.xy',
                                    name='tester',
                                    xmlname='tests/data/sample_db_uc.xml',
                                    imname='tests/data/sample_mask.tif',
                                    lx=500, ly=500)
        
    @nose.tools.timed(3)
    def test_set_features_timing(self):
        self.matcher_uc.set_features('test')

