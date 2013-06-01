"""
@package test_coord
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Coord
"""

import math
import nose.tools

from picolo.config import Coord

class TestCoord:

    def test_init_zero_angle(self):
        c = Coord(0,0)
        nose.tools.assert_almost_equals(c.theta, 0)
        
    def test_init_zero_length(self):
        c = Coord(0,0)
        nose.tools.assert_almost_equals(c.r, 0)   
    
    def test_add(self):
        c1 = Coord(0,1)
        c2 = Coord(2,3)
        csum = c1 + c2
        nose.tools.assert_almost_equals(csum.x, 2)
        nose.tools.assert_almost_equals(csum.y, 4)
        
    def test_subtract(self):
        c1 = Coord(3,1)
        c2 = Coord(2,3)
        cdiff = c1 - c2
        nose.tools.assert_almost_equals(cdiff.x, 1)
        nose.tools.assert_almost_equals(cdiff.y, -2)
        nose.tools.assert_almost_equals(cdiff.r, math.sqrt(5))
        
    def test_radians_in_range_neg(self):
        c = Coord(-1,-1)
        nose.tools.assert_almost_equals(c.theta,-math.pi * 3.0 / 4.0)
        
    def test_radians_in_range_pos(self):
        c = Coord(-1,1)
        nose.tools.assert_almost_equals(c.theta, math.pi * 3.0 / 4.0)
        
    def test_degrees_in_range_neg(self):
        c = Coord(-1,-1)
        nose.tools.assert_almost_equals(c.degrees, -180+45)
        
    def test_degrees_in_range_pos(self):
        c = Coord(-1,1)
        nose.tools.assert_almost_equals(c.degrees, 180-45)
        
    def test_repr(self):
        nose.tools.assert_equal(repr(Coord(0,1)),
                                "(x,y) = (0.0, 1.0); (r,theta) = (1.0, 90.0)")
                                
    def test_equal(self):
        nose.tools.assert_true(Coord(1,0) == Coord(1,0))                           
                                
    def test_rotate(self):
        c = Coord(2,1)
        r = c.rotate(math.radians(90))
        nose.tools.assert_equal(r, Coord(-1,2))