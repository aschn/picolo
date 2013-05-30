"""
@package test_config
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Config
"""

import numpy as np
import nose.tools

from picolo.config import Config, Coord, Mask

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
        
        self.infile = 'tests/data/sample_mask.tif'
        self.mask = Mask(self.infile, self.Lx, self.Ly)

    def test_init_default_params(self):
        nose.tools.assert_false(self.config_default.doPBC)
        nose.tools.assert_equal(self.config_default.Lx, max(self.xcoords))
        nose.tools.assert_equal(self.config_default.Ly, max(self.ycoords))
        
    def test_init_default_vals(self):
        nose.tools.assert_equal(self.config_default.N, len(self.xcoords))
        nose.tools.assert_true(np.all(self.xcoords == self.config_nopbc.x))
        nose.tools.assert_true(np.all(self.ycoords == self.config_nopbc.y))
        
    @nose.tools.raises(ValueError)
    def test_bad_init_Lx(self):
        bad_config = Config(self.xcoords, self.ycoords, pbc=False,
                            lx=(max(self.xcoords)-1), ly=self.Ly)
    @nose.tools.raises(ValueError)
    def test_bad_init_Ly(self):
        bad_config = Config(self.xcoords, self.ycoords, pbc=False,
                            lx=self.Lx, ly=(max(self.ycoords)-1))
    @nose.tools.raises(ValueError)
    def test_bad_init_xc(self):
        bad_config = Config([2, 2], [-1, 5], pbc=False,
                            lx=self.Lx, ly=self.Ly)
    @nose.tools.raises(ValueError)
    def test_bad_init_yc(self):
        bad_config = Config([-1, 5], [2, 2], pbc=False,
                            lx=self.Lx, ly=self.Ly)
   
    def test_init_nopbc(self):
        nose.tools.assert_true(np.all(self.config_nopbc.x == self.config_nopbc.ximages))
        nose.tools.assert_true(np.all(self.config_nopbc.y == self.config_nopbc.yimages))
        nose.tools.assert_equal(self.config_nopbc.Lx, self.Lx)
        nose.tools.assert_equal(self.config_nopbc.Ly, self.Ly)

    def test_init_pbc(self):
        nose.tools.assert_true(self.config_pbc.doPBC)
        nose.tools.assert_equal(len(self.config_pbc.x)*9,
                                len(self.config_pbc.ximages))
        nose.tools.assert_equal(len(self.config_pbc.y)*9,
                                len(self.config_pbc.yimages))
        nose.tools.assert_equal(self.config_pbc.Lx, self.Lx)
        nose.tools.assert_equal(self.config_pbc.Ly, self.Ly)
        
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
        nose.tools.assert_true(self.config_nopbc.is_tri_in_image(0, 1, 2))
        nose.tools.assert_true(self.config_pbc.is_tri_in_image(0, 1, 2))
        nose.tools.assert_true(self.config_pbc.is_tri_in_image(0, 1, 2+4*len(self.config_pbc.x)))
        nose.tools.assert_false(self.config_pbc.is_tri_in_image(0, 1, 2+5*len(self.config_pbc.x)))
        
    def test_coord_at(self):
        expected_coord = Coord(self.xcoords[0], self.ycoords[0])
        actual_coord = self.config_default.coord_at(0)
        for attr in ['x', 'y', 'r', 'theta']:
            nose.tools.assert_almost_equals(getattr(expected_coord, attr),
                                            getattr(actual_coord, attr))
                
    def test_density(self):
        default_density = float(len(self.xcoords)) / max(self.xcoords) / max(self.ycoords)
        set_box_density = float(len(self.xcoords)) / self.Lx / self.Ly
        masked_box_density = float(len(self.xcoords)) / self.mask.area()
        nose.tools.assert_almost_equals(self.config_default.density(),
                                        default_density)
        nose.tools.assert_almost_equals(self.config_nopbc.density(),
                                        set_box_density)
        nose.tools.assert_almost_equals(self.config_pbc.density(),
                                        set_box_density)
        nose.tools.assert_almost_equals(self.config_nopbc.density(self.mask, cutoff_dist=0),
                                        masked_box_density)
                                        
 #   def test_interior(self):
 #       nose.tools.assert_equal(self.config_nopbc.interior_particles(self.mask), 5)
 #       nose.tools.assert_equal(self.config_nopbc.interior_particles(self.mask, 0.2), 10)

    def test_radial(self):
        gr_nopbc = self.config_nopbc.radial_distribution()
        gr_pbc = self.config_pbc.radial_distribution()
        
    def test_nnd(self):
        nnd_nopbc = self.config_nopbc.nearest_neighbor_distribution()
        nnd_pbc = self.config_pbc.nearest_neighbor_distribution()