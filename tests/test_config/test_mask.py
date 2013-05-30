"""
@package test_mask
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Mask
"""

import nose.tools

from picolo.config import Mask

class TestMask:
    
    def setup(self):
        self.Lx = 500
        self.Ly = 500
        self.infile = 'tests/data/sample_mask.tif'
        self.mask = Mask(self.infile, self.Lx, self.Ly)
        
    def test_init(self):
        nose.tools.assert_almost_equals(self.mask.cf_px2nm, 1)        
        nose.tools.assert_almost_equals(self.mask.cf_nm2px, 1)        
        nose.tools.assert_equal(self.mask.extent, (0, self.Lx, 0, self.Ly))
        
    def test_empty_init(self):
        default_mask = Mask()
        size = default_mask._mask.shape[0] * default_mask._mask.shape[0]
        nvalid = sum(default_mask._mask)
        nose.tools.assert_equal(size, nvalid)
        
    def test_nm2px(self):
        nose.tools.assert_equal((0,self.mask._mask.shape[1]-1),
                                self.mask._point_nm2px(0, self.Ly-1e-5))
        nose.tools.assert_equal((0,0), self.mask._point_nm2px(0, 0))

    def test_mask(self):
        nose.tools.assert_false(self.mask._mask[1,1])
        nose.tools.assert_true(self.mask._mask[400,200])
        nose.tools.assert_false(self.mask._mask[200,400])
        
    def test_area(self):
        nose.tools.assert_almost_equals(self.mask.area(), 107067)
        nose.tools.assert_almost_equals(self.mask.area(0), 107067)
        nose.tools.assert_less(self.mask.area(10), 107067)
        
    def tecdst_valid_seg(self):
        nose.tools.assert_true(self.mask.is_valid_seg(400, 200, 400, 300))
        nose.tools.assert_false(self.mask.is_valid_seg(400, 200, 200, 200))
        
    def test_interior(self):
        nose.tools.assert_true(self.mask.is_interior(400, 200, 79))
        nose.tools.assert_false(self.mask.is_interior(400, 200, 80))
        
    def test_dist_to_edge(self):
        nose.tools.assert_almost_equals(self.mask.dist_to_edge(400, 200),
                                        79.0759128939)
                                        