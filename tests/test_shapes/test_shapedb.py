"""
@package test_shapedb
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.ShapeDB
"""

import nose.tools
import logging

from picolo.shapes.shapedb import ShapeDB
from picolo.shapes.shapes import shape_factory_from_values
from picolo.shapes.classifiers import classifier_factory

class TestShapeDB:

    def setup(self):
        logging.basicConfig(level=logging.DEBUG)
        self.db_default = ShapeDB()
        self.s = shape_factory_from_values('generic', ['a'], [2],
                                           {'int':4, 'str':'b', 'csv':[1,3,6]})
        self.db_one = ShapeDB()
        self.db_one.add('test', self.s)
        
    def test_init(self):
        nose.tools.assert_equal(self.db_default.null_shape_name, '')
        nose.tools.assert_equal(self.db_default._classifier.algorithm(), 'default')

    def test_len(self):
        nose.tools.assert_equal(len(self.db_default), 0)        
        
    @nose.tools.raises(KeyError)
    def test_getitem(self):
        self.db_default['test']
        
    def test_add(self):
        self.db_default.add('test', self.s)
        nose.tools.assert_equal(len(self.db_default), 1)
        
    def test_shape_type(self):
        nose.tools.assert_equal(self.db_one.shape_type(), 'Generic')
        nose.tools.assert_equal(self.db_default.shape_type(), 'Generic')
        
    def test_classnames(self):
        nose.tools.assert_equal(self.db_default.class_names(), [''])
        nose.tools.assert_equal(self.db_one.class_names(), ['', 'test'])
        
    def test_names(self):
        nose.tools.assert_equal(self.db_default.names(), [])
        nose.tools.assert_equal(self.db_one.names(), ['test'])
     
    def test_discard(self):
        self.db_one.discard('test')
        nose.tools.assert_equal(len(self.db_one), 0)   
        nose.tools.assert_equal(len(self.db_one.names()), 0)   
        nose.tools.assert_equal(len(self.db_one.class_names()), 1)   
        
    def test_match(self):
        matchval = self.db_one.compute_match('test', self.s)
        expected = classifier_factory().compute_match(self.db_one['test'], self.s)
        ismatch = self.db_one.is_match('test', matchval)
        nose.tools.assert_almost_equal(matchval, expected)
        nose.tools.assert_false(ismatch)
        
    def test_save(self):
        returncode = self.db_one.save('tests/data/sample_db_out.xml')
        nose.tools.assert_equal(returncode, 1)
        
    def test_load(self):
        self.db_default.load('tests/data/sample_db_in.xml')
        nose.tools.assert_true('test' in self.db_default.names())

    def test_load_uc(self):
        self.db_default.load('tests/data/sample_db_uc.xml')
        nose.tools.assert_true('test' in self.db_default.names())
        nose.tools.assert_equal(self.db_default._classifier.algorithm(),
                                'GMM')
        for var in ['a', 'b', 'degrees']:
            nose.tools.assert_true(self.db_default['test'].has_component(var))
        nose.tools.assert_greater(self.db_default._classifier._rejection_cutoff,
                                  0)
        nose.tools.assert_true(self.db_default['test'].get('is_valid'))
                                  
    def test_match_uc(self):
        self.db_default.load('tests/data/sample_db_uc.xml')
        shape = self.db_default['test']
        if not shape.get('is_valid'):
            shape.put_param('is_valid', True)
        val = self.db_default.compute_match('test', shape)
        nose.tools.assert_almost_equal(val, 0.021164545311413662)
        ismatch = self.db_default.is_match('test', val)
        nose.tools.assert_true(ismatch)

    def test_init_w_file(self):
        sdb = ShapeDB('tests/data/sample_db_in.xml')
        nose.tools.assert_true(sdb.names(), ['test'])
