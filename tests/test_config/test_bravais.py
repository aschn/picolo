"""
@package test_bravais
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Bravais
"""

import nose.tools
import math

from picolo.config import Coord, BravaisLattice

class TestBravais:
    
    def setup(self):
        self.coords = [Coord(15,0), Coord(0,16), Coord(-15,0), Coord(0,-16)]
        self.xy_tuple = (self.coords[0].x, self.coords[0].y,
                            self.coords[1].x, self.coords[1].y)
        self.max_dist = 30.0
        self.min_dist = 14.0
        self.rcut = 7.0

        self.bl = BravaisLattice()
    
    def test_init(self):
        nose.tools.assert_true(self.bl)
        
    def test_kl(self):
        n_kls = 2
        kls = self.bl.kl_pairs(n_kls)
        nose.tools.assert_equal(kls.shape[0], (n_kls * 2 + 1)**2)
        nose.tools.assert_equal(kls.shape[1], 2)
        
    def test_expand(self):
        n_kls = 1
        ps = self.bl.expand(*self.xy_tuple, n_kls=n_kls)
        nose.tools.assert_equal(len(ps), (n_kls) * (n_kls * 2 + 1)**2)
        for c in self.coords:
            count = 0
            for p in ps:
                if c == p:
                    count += 1
            nose.tools.assert_equal(count, 1)
            
    def test_error_exact(self):
        error = self.bl.error(self.xy_tuple, self.coords, 2, self.rcut)
        nose.tools.assert_almost_equal(error, 0)
        error = self.bl.error(self.xy_tuple, [Coord(15,16)], 1, self.rcut)
        nose.tools.assert_almost_equal(error, 0.0)
         
    def test_error_far(self):
        error = self.bl.error(self.xy_tuple, [Coord(22,16)], 1, self.rcut)
        nose.tools.assert_almost_equal(error, 1)
        
    @nose.tools.timed(0.005)
    def test_fit(self):
        fitted_bl, error = self.bl.fit(self.coords, 1, self.rcut,
                                self.max_dist, self.min_dist)
        print fitted_bl
        for c in self.coords:
            count = 0
            for p in fitted_bl:
                print c.rotate(math.radians(90)), p
                if c.rotate(math.radians(90)) == p:
                    count += 1
            nose.tools.assert_equal(count, 1)
            