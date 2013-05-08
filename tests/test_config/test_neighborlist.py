"""
@package test_neighborlist
@author Anna Schneider
@version 0.1
@brief Tests picolo.config.NeighborList
"""

import nose.tools
import numpy as np

from picolo.config.neighborlist import NeighborList
from picolo.config import Mask, Config, DelaunayNeighbors, DistNeighbors

class TestNeighborList:
    
    def setup(self):
        Lx = 500
        Ly = 500

        # set up config
        coords = np.genfromtxt('tests/data/sample_config.xy')
        config = Config(coords[:,0], coords[:,1],
                        pbc=False, lx=Lx, ly=Ly)
                     
        # set up mask
        infile = 'tests/data/sample_mask.tif'
        mask = Mask(infile, Lx, Ly)

        self.neighborlist_default = NeighborList(config)
        self.neighborlist_masked = NeighborList(config, mask)
        self.neighborlist_delaunay = DelaunayNeighbors(config, mask)
        self.neighborlist_dist = DistNeighbors(config, mask=mask, dist=30)
        
    def test_init(self):
        assert self.neighborlist_default.is_masked == False
        assert self.neighborlist_masked.is_masked == True
        
    def test_len(self):
        assert len(self.neighborlist_default) == 0
        assert len(self.neighborlist_masked) == 0
        assert len(self.neighborlist_delaunay) == 209
        assert len(self.neighborlist_dist) == 209

    def test_neighbors_of(self):
        nof_default = self.neighborlist_default.neighbors_of(0)
        nof_masked = self.neighborlist_masked.neighbors_of(0)
        nof_delaunay = self.neighborlist_delaunay.neighbors_of(0)
        nof_dist = self.neighborlist_dist.neighbors_of(0)
        assert nof_default == set()
        assert nof_masked == set()
        assert nof_delaunay == set([40, 45, 179, 85])
        assert nof_dist == set([179, 85])
        
    def test_are_neighbors(self):
        for i in self.neighborlist_default.neighbors_of(0):
            assert self.neighborlist_default.are_neighbors(0, i)
        
        for i in self.neighborlist_masked.neighbors_of(0):
            assert self.neighborlist_masked.are_neighbors(0, i)

        for i in self.neighborlist_delaunay.neighbors_of(0):
            assert self.neighborlist_delaunay.are_neighbors(0, i)

        for i in self.neighborlist_dist.neighbors_of(0):
            assert self.neighborlist_dist.are_neighbors(0, i)

        assert self.neighborlist_delaunay.are_neighbors(0,0) == False        
        
    def test_iteritems(self):
        count = 0
        for i, nids in self.neighborlist_delaunay.iteritems():
            count += 1
            assert nids == self.neighborlist_delaunay.neighbors_of(i)
            
        assert count == len(self.neighborlist_delaunay)
        
    def test_kNN(self):
        all_ones = [1 for i in range(len(self.neighborlist_dist))]
        all_ones_but_one = [0] + [1 for i in range(len(self.neighborlist_dist)-1)]
        
        assert all_ones == self.neighborlist_dist.kNN_filter(all_ones)
        assert all_ones == self.neighborlist_dist.kNN_filter(all_ones_but_one)
        assert all_ones == self.neighborlist_dist.kNN_filter(all_ones_but_one, mode='median')
        
        nn_of_0 = float(len(self.neighborlist_dist.neighbors_of(0)))
        expected = (nn_of_0) / (nn_of_0 + 1)
        results = self.neighborlist_dist.kNN_filter(all_ones_but_one, mode='mean')
        nose.tools.assert_almost_equals(results[0], expected)
        
        