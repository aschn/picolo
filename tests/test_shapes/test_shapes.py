"""
@package test_shapes
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.Shape
"""

import nose.tools
import math

from picolo.shapes.shapes import Shape, FourierShape, ZernikeShape, UnitCellShape
from picolo.shapes.shapes import shape_factory_from_coords, shape_factory_from_values
from picolo.config import Coord

class TestBaseShape:

    def setup(self):
        self.shape_empty = Shape()
        
        self.vars = ['a', 3, 'b']
        self.vals = range(3)
        self.shape_full = Shape(self.vars, self.vals)
        
    def test_factory_from_vals(self):
        s = shape_factory_from_values('generic', self.vars, self.vals,
                                      {'test': True})
        nose.tools.assert_equal(s.get('test'), True)
        for var, val in zip(self.vars, self.vals):
            nose.tools.assert_almost_equal(s.get(var), val)
        
    @nose.tools.raises(ValueError)
    def test_bad_init(self):
        Shape(self.vars, range(len(self.vars)+1))
        
    def test_good_init(self):
        nose.tools.assert_equal(self.shape_empty.get('type'), 'Generic')
        
    def test_len(self):
        nose.tools.assert_equal(len(self.shape_empty), 0)
        nose.tools.assert_equal(len(self.shape_full), len(self.vars))
        
    def test_get(self):
        for var, val in zip(self.vars, self.vals):
            nose.tools.assert_almost_equal(self.shape_full.get(var), val)
            
    @nose.tools.raises(KeyError)
    def test_get_raises(self):
        self.shape_empty.get('a')
        
    def test_get_vals(self):
        for i, val in enumerate(self.shape_full.get_vals()):
            nose.tools.assert_almost_equal(self.vals[i], val)
            
    def test_norm(self):
        for i, val in enumerate(self.shape_full.get_vals(norm=True)):
            nose.tools.assert_almost_equal(self.vals[i]/self.shape_full.mag(), val)
                    
    def test_iter_components(self):
        for var, val in self.shape_full.iter_components():
            nose.tools.assert_almost_equal(val, self.vals[self.vars.index(var)])

    def test_iter_params(self):
        for var, val in self.shape_full.iter_params():
            nose.tools.assert_almost_equal(val, self.shape_full.get(var))

    def test_mag(self):
        mag = math.sqrt(sum([x*x for x in self.vals]))
        nose.tools.assert_almost_equal(self.shape_full.mag(), mag)

    def test_has_component(self):
        nose.tools.assert_true(self.shape_full.has_component(3))
        nose.tools.assert_false(self.shape_full.has_component(1))
        
    def test_drop_component(self):
        nose.tools.assert_true(self.shape_full.has_component(3))
        self.shape_full.drop_component(3)
        nose.tools.assert_false(self.shape_full.has_component(3))
        
    def test_put_component(self):
        self.shape_empty.put_component(1, 1)
        nose.tools.assert_almost_equal(self.shape_empty.get(1), 1)
        
    def test_put_param(self):
        self.shape_empty.put_param('c', 5)
        self.shape_full.put_param(4, 5)
        nose.tools.assert_equal(self.shape_empty.get('c'), 5)
        nose.tools.assert_equal(self.shape_full.get(4), 5)
        
    def test_subset(self):
        subsetted = self.shape_full.subset(self.vars[:1])
        subsetted_data = [(var, val) for var, val in subsetted.iter_components()]           
        from_scratch = Shape(self.vars[:1], self.vals[:1])
        scratch_data = [(var, val) for var, val in from_scratch.iter_components()]                
        nose.tools.assert_almost_equal(subsetted_data, scratch_data)
        
class TestRealShapes:
    
    def setup(self):
        self.coords = [Coord(15,0), Coord(0,15), Coord(-15,0), Coord(0,-15)]

        self.uc = UnitCellShape(neighbor_dist=30)
        self.fourier = FourierShape()
        self.zernike = ZernikeShape(neighbor_dist=30)
    
    def test_uc_factory(self):
        factory_shape = shape_factory_from_coords(self.coords, self.uc)
        factory_data = [(var, val) for var, val in factory_shape.iter_components()]
        self.uc.build_from_coords(self.coords)
        built_data = [(var, val) for var, val in self.uc.iter_components()]
        nose.tools.assert_equal(factory_data, built_data)
        
    def test_uc_data(self):
        self.uc.build_from_coords(self.coords)
        nose.tools.assert_almost_equal(self.uc.get('a'), 15.0, places=4)
        nose.tools.assert_almost_equal(self.uc.get('b'), 15.0, places=4)
        nose.tools.assert_almost_equal(self.uc.get('degrees'), 90.0, places=3)
        nose.tools.assert_almost_equal(self.uc.get('theta'),
                                       math.radians(90.0), places=4)
        nose.tools.assert_almost_equal(self.uc.area(), 15.0*15.0, places=4)
        nose.tools.assert_equal(self.uc.get('type'), 'UnitCell')
        
    def test_fourier_factory(self):
        factory_shape = shape_factory_from_coords(self.coords, self.fourier)
        factory_data = [(var, val) for var, val in factory_shape.iter_components()]
        self.fourier.build_from_coords(self.coords)
        built_data = [(var, val) for var, val in self.fourier.iter_components()]
        nose.tools.assert_equal(factory_data, built_data)
        
    def test_fourier_data(self):
        self.fourier.build_from_coords(self.coords)
        for var, val in self.fourier.iter_components():
            if var % 4 == 0:
                nose.tools.assert_almost_equal(val, 1)
            else:
                nose.tools.assert_almost_equal(val, 0)
        nose.tools.assert_equal(self.fourier.get('type'), 'Fourier')

    def test_zernike_factory(self):
        factory_shape = shape_factory_from_coords(self.coords, self.zernike)
        factory_data = [(var, val) for var, val in factory_shape.iter_components()]
        self.zernike.build_from_coords(self.coords)
        built_data = [(var, val) for var, val in self.zernike.iter_components()]
        nose.tools.assert_equal(factory_data, built_data)
        
    def test_zernike_data(self):
        self.zernike.build_from_coords(self.coords)
        for var, val in self.zernike.iter_components():
            if var[1] % 4 == 0:
                nose.tools.assert_greater(val, 0)
            else:
                nose.tools.assert_almost_equal(val, 0)
        nose.tools.assert_equal(self.zernike.get('type'), 'Zernike')
