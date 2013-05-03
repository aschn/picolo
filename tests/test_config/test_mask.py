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
        assert self.mask.extent == (0, self.Lx, 0, self.Ly)
        assert self.mask.edgefile == 'tests/data/sample_mask_pxToEdge.txt'
        
    def test_nm2px(self):
        assert (0,self.mask.mask.shape[1]-1) == self.mask._point_nm2px(0, self.Ly-1e-5)
        assert (0,0) == self.mask._point_nm2px(0, 0)

    def test_mask(self):
        assert self.mask.mask[1,1] == False
        assert self.mask.mask[400,200] == True
        assert self.mask.mask[200,400] == False
        
    def test_area(self):
        nose.tools.assert_almost_equals(self.mask.get_area(), 107067)
        
    def test_valid_seg(self):
        assert self.mask.is_valid_seg(400, 200, 400, 300) == True
        assert self.mask.is_valid_seg(400, 200, 200, 200) == False
        
    def test_interior(self):
        assert self.mask.is_interior(400, 200, 79) == True
        assert self.mask.is_interior(400, 200, 80) == False
        
    def test_dist_to_edge(self):
        nose.tools.assert_almost_equals(self.mask.dist_to_edge(400, 200),
                                        79.0759128939)