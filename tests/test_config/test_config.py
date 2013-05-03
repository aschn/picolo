"""
@package test_config
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Config
"""

import numpy as np
import nose.tools

from picolo.config import Config, Coord

class TestConfig:
    
    def setup(self):
        self.xcoords = [2, 2, 8, 8, 5]
        self.ycoords = [2, 8, 2, 8, 5]
        self.Lx = 10.0
        self.Ly = 10.0
        
        self.config_default = Config(self.xcoords, self.ycoords)
        self.config_pbc = Config(self.xcoords, self.ycoords,
                                 pbc=True, lx=self.Lx, ly=self.Ly)
        self.config_nopbc = Config(self.xcoords, self.ycoords,
                                   pbc=False, lx=self.Lx, ly=self.Ly)
        
    def test_init_default_params(self):
        assert self.config_default.doPBC == False
        assert self.config_default.Lx == max(self.xcoords)
        assert self.config_default.Ly == max(self.ycoords)
        
    def test_init_default_vals(self):
        assert self.config_default.N == len(self.xcoords)
        assert np.all(self.xcoords == self.config_nopbc.x)
        assert np.all(self.ycoords == self.config_nopbc.y)
   
    def test_init_nopbc(self):
        assert np.all(self.config_nopbc.x == self.config_nopbc.ximages)
        assert np.all(self.config_nopbc.y == self.config_nopbc.yimages)
        assert self.config_nopbc.Lx == self.Lx
        assert self.config_nopbc.Ly == self.Lx

    def test_init_pbc(self):
        assert self.config_pbc.doPBC == True
        assert len(self.config_pbc.x)*9 == len(self.config_pbc.ximages)
        assert len(self.config_pbc.y)*9 == len(self.config_pbc.yimages)
        assert self.config_pbc.Lx == self.Lx
        assert self.config_pbc.Ly == self.Lx
        
    def test_pbc_dist(self):
        xdist_w_pbc = self.config_pbc._PBC_dist(2, 5, 8, 5)
        xdist_wo_pbc = self.config_nopbc._PBC_dist(2, 5, 8, 5)
        ydist_w_pbc = self.config_pbc._PBC_dist(5, 2, 5, 8)
        ydist_wo_pbc = self.config_nopbc._PBC_dist(5, 2, 5, 8)
        nose.tools.assert_almost_equals(xdist_w_pbc, 4)
        nose.tools.assert_almost_equals(xdist_wo_pbc, 6)
        nose.tools.assert_almost_equals(ydist_w_pbc, 4)
        nose.tools.assert_almost_equals(ydist_wo_pbc, 6)
       
    def test_tri_in_image(self):
        assert self.config_nopbc.is_tri_in_image(0, 1, 2)
        assert self.config_pbc.is_tri_in_image(0, 1, 2)
        assert self.config_pbc.is_tri_in_image(0, 1, 2+4*len(self.config_pbc.x))
        assert self.config_pbc.is_tri_in_image(0, 1, 2+5*len(self.config_pbc.x)) is False
        
    def test_coord_at(self):
        expected_coord = Coord(self.xcoords[0], self.ycoords[0])
        actual_coord = self.config_default.coord_at(0)
        for attr in ['x', 'y', 'r', 'theta']:
            nose.tools.assert_almost_equals(getattr(expected_coord, attr),
                                            getattr(actual_coord, attr))
                
    def test_density(self):
        default_density = float(len(self.xcoords)) / max(self.xcoords) / max(self.ycoords)
        set_box_density = float(len(self.xcoords)) / self.Lx / self.Ly
        nose.tools.assert_almost_equals(self.config_default.density(), default_density)
        nose.tools.assert_almost_equals(self.config_nopbc.density(), set_box_density)
        nose.tools.assert_almost_equals(self.config_pbc.density(), set_box_density)
