"""
@package config
@module neighborlist
@author Anna Schneider
Contains classes for NeighborList, DistNeighbors, DelaunayNeighbors
"""

# import from standard library
import sys
import itertools as it

# import external packages
import numpy as np
from scipy import spatial
import matplotlib.delaunay as delaunay

class NeighborList:
    """ Base class for classes that stores nearest neighbors of points.
        Provides a dict-like interface.
        For a useful implementation, override compute(config) method,
            and possibly _check(config, mask) and apply_mask(config, mask)
        
    Attributes:
        
    @var _neighbor_dict dict with key = particle id,
        val = set of neighbor particle ids
        
    @var is_masked Bool for whether or not mask has been applied
    
    @var _ids Particle ids that have a set of neighbors (may be empty)
        
    """
    
    def __init__(self, config, mask=None):    
        """ Constructor.
        
        @param self The object pointer     
        
        @param config Config object
        
        @param mask Mask object, default is None
        
        """
        self._build(config, mask)
    
    def _build(self, config, mask):
        """ Initialized data based on config and mask.
        
        @param self The object pointer     
        
        @param config Config object
        
        @param mask Mask object, default is None
        
        """
        # initialize 
        self._neighbor_dict = self.compute(config)
        if mask is not None:
            self.apply_mask(config, mask)
            self.is_masked = True
        else:
            self.is_masked = False
        self._check(config, mask)
        self._ids = self._neighbor_dict.keys()

    def __getitem__(self, i):
        """ Like dict.__getitem__ 
        """
        if i in self._neighbor_dict:
            return self._neighbor_dict[i] 
        else:
            return []
    def __len__(self):
        """ Like dict.__repr__ 
        """
        return len(self._neighbor_dict)  
    def __repr__(self):
        """ Like dict.__repr__ 
        """
        return repr(self._neighbor_dict)
    def iteritems(self):
        """ Like dict.iteritems
        """        
        return self._neighbor_dict.iteritems()
        
    def neighbors_of(self, i):
        """ Alias for __getitem__
        """
        return self.__getitem__(i)
        
    def compute(self, config):
        """ Figures out connections between particles in a config.
            Should be implemented in every derived class.
        
        @param self The object pointer
        
        @param config Config object
        
        @retval dict with key = particle id,
            val = set of neighbor particle ids
            
        """
        pass
    def _check(self, config, mask):
        """ Checks the connections.
            Should be implemented in some derived classes.
        
        @param self The object pointer
        
        @param config Config object
        
        @retval dict with key = particle id,
            val = set of neighbor particle ids
            
        """
        pass
    def apply_mask(self, config, mask):
        """ Remove links that cross invalid regions of the mask.
            May need to be overridden in some derived classes.
        
        @param self The object pointer
        
        @param config Config object
        
        @param mask Mask object
        
        """
        # check for mask
        if not mask:
            return

        # set up new copies of triangles and neighbors
        masked_neighbors = self._neighbor_dict.copy()

        # loop over points
        for ip1 in self._ids:
            # get coords for point
            x1, y1 = self.config.ximages[ip1], self.config.yimages[ip1]
            
            # loop over neighbors
            for ip2 in self.neighbors_of(ip1):
                # get coords for neighbor
                x2, y2 = self.config.ximages[ip2], self.config.yimages[ip2]
                
                
                # check if edge crosses mask boundary
                if not mask.is_valid_seg(x1, y1, x2, y2):

                    # if invalid, remove from masked_triangles and neighbors
                    # use discard (doesn't raise error if missing)
                    #    in case we've seen this pair before
                    masked_neighbors[ip1].discard(ip2)
                    masked_neighbors[ip2].discard(ip1)

        # finish
        self._neighbor_dict = masked_neighbors
                
    def kNN_filter(self, vals, inds=None, ownweight=1, mode='mode'):
        """ Implements a k-nearest-neighbors filter on the data in vals
            using the neighbor definitions in the neighbor list.
            If mode = 'median' or 'mean', select the median or mean value
                over the neigbhors (good for continuous data).
            If mode = 'mode', select the most common value over the
                neighbors (good for categorical data).
        
        @param self The object pointer
        
        @param vals Iterable of data to filter
        
        @param inds Iterable of ints for which indices to consider, 
            default is to use all.
            
        @param ownweight Switch to create a weighted filter. Options are
            any integer (counts self that many times),
            'half' (counts self half the number of neighbors times), or
            'n-1' (counts self as the number of neighbors minus 1).
            Default is 1.
            
        @param mode Switch to change the filter method. Default is 'mode'.
        
        """                                                  
        # set up storage for new filtered vals
        newvals = []
        
        # set up inds
        if not inds:
            inds = self._ids
            
        # loop over ids
        for ipv, ip in enumerate(inds):
            # start weighted list
            if ownweight == 'half':
                ownweight = len(self.neighbors_of(ip)/2
            elif ownweight == 'n-1':
                ownweight = len(self.neighbors_of(ip)) - 1
            else:
                ownweight = int(ownweight)
            if ownweight < 1:
                ownweight = 1
            thisvals = [vals[ipv] for j in range(ownweight)]
            
            # add neighbor vals to list
            for ip2 in self.neighbors_of(ip):
                if ip2 in inds:
                    ipv2 = inds.index(ip2)
                    thisvals.append(vals[ipv2])
            
            if mode == 'mode':
                # choose most common
                try:
                    newvals.append( np.argmax(np.bincount(thisvals)) )
                except:
                    print ip, len(vals)
                    
            elif mode == 'mean':
                # choose average
                newvals.append(np.mean(thisvals))
                
            elif mode == 'median':
                # use self for tie breaker if even number
                if len(thisvals) % 2 == 0:
                    thisvals.append(vals[ipv])
                    
                # choose median
                newvals.append(np.median(thisvals))
              #  print np.median(thisvals), sorted(thisvals)
            else:
                print "ERROR: invalid mode", mode, "in majorityRule"
                raise
                
        # return
        return newvals
        
class DistNeighbors(NeighborList):

    def __init__(self, config, dist = 0, mask = None):
        """ Constructor.
        
        @param self The object pointer     
        """
        # initialize
        self._kdtree = spatial.KDTree(zip(config.x, config.y)) 
        self._r = dist
        self._build(config, mask)
   
    def compute(self, config=None):
        # make neighbor list with cutoff
        neighbor_list = self._kdtree.query_ball_tree(self._kdtree, self._r)

        # prune while making dict
        self._neighbor_dict = dict()
        for ip, neighbors in enumerate(neighbor_list):
            # remove self
            neighbors.remove(ip)
                                
            # add to dict
            self._neighbor_dict[ip] = neighbors
                                      
    def neighbors_within(self, dist, ip):
        # set up with correct distance
        if dist != self._r:
            self._r = dist
            self.compute()
            
        # return 
        return self.neighbors_of(ip)

####################################################

class DelaunayNeighbors(NeighborList):
    
    def __init__(self, config, mask = None):
        """ Constructor.
        
        @param self The object pointer     
        
        @param config Config object
        
        @param mask Mask object, default is None
        
        """
        self._build(config, mask)
        
    def compute(self, config):
        """ Calculate the Delaunay triangulation.
            Sets triangles, a set of 3-tuples of particle ids in
                config.*images that form the vertices of a Delaunay triangle.

        @param self The object pointer
        
        @param config Config object
        
        @retval neighbors Dict with key=vertex,
            val=set of particle ids in *coords based on Delaunay triangulation

        """
        # use matplotlib's Delaunay triangulation
        # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/tri/triangulation.py
        # centers is Ntri x 2 float array of circumcenter x,y coords (ie vertices in Voronoi)
        # edges is N? x 2 int array of indices defining triangle edges
        # triangles is Ntri x 3 int array of triangle vertex indices, ordered counterclockwise
        # tri_nhbrs is Ntri x 3 int array of triangle indices that share edges (null = -1)
        #   tri_nhbrs[i,j] is the triangle that is the neighbor to the edge from
        #   point index triangles[i,j] to point index triangles[i,(j+1)%3].
        centers, edges, triangles_list, tri_nbhrs = delaunay.delaunay(config.ximages,
                                                                      config.yimages)

        # store triangles as set of tuples, only keeping those in or near
        triangles_set = set()
        for tri in triangles_list:
            if config.is_tri_in_image(tri[0], tri[1], tri[2]):
                triangles_set.add(tuple(sorted(tri)))

        # identify nearest neighbors in triangulation
        # nbhrs is a dict with key=vertex, val=set of neighbor particle ids
        # all particle indices are in original image
        nbhrs = dict(zip(config.indices, [set() for i in config.indices]))
        for tri in triangles_set:
            inds = [ip % int(config.N) for ip in tri]
            for ivert in range(3): # loop through triangle vertices
                for offset in [1,2]: # loop through the two neighboring vertices
                    nbhrs[inds[ivert]].add(inds[(ivert+offset)%3])

        # return neighbors and triangles
        self._triangles = triangles_set
        return nbhrs

    def _check(self, config, mask):
        """ Check that Delaunay triangulation is valid.
        
        @param self The object pointer     

        @param config Config object

        @param mask Mask object
        
        """
        # check number of triangles (only known for PBC)
        if config.doPBC:
            try:
                assert(len(self._triangles) == 2 * config.N)
            except AssertionError:
                print "***"
                print "ERROR in number of Delaunay triangles with PBC"
                print "got %d triangles, should have %d" % (len(self._triangles),
                                                            2 * config.N)
                sys.exit()

        # check that every particle has neighbors
        try:
            assert(len(self._ids) == config.N)
        except AssertionError:
            print "***"
            print "ERROR in number of particles with neighbors"
            print "got %d particles with neighbors, should have %d" % (len(self.neighbor_dict.keys()), config.N)
            sys.exit()
                   
        # assemble histogram of neighbors
        # counts is dict with key = number of neighbors, value = number of vertices with that many neighbors
        counts = dict()
        for ip in range(config.N):
            nnbhrs = len(self.neighbors_of(ip))
            if nnbhrs == 0 and mask is not None:
                print "particle %d has 0 neighbors, remove it from neighbor list" % ip
                del self._neighbor_dict[ip]
            elif nnbhrs in counts:
                counts[nnbhrs] += 1
            else:
                counts[nnbhrs] = 1
        print "nbhr histogram:", counts, sum(counts.values())

        # check histogram
        if not mask:
            try:
                assert(0 not in counts)
            except AssertionError:
                print "***"
                print "ERROR in number of neighboring particles"
                print "got %d particles with 0 neighbor" % (counts[0])
                sys.exit()
            try:
                assert(1 not in counts)
            except AssertionError:
                print "***"
                print "ERROR in number of neighboring particles"
                print "got %d particles with 1 neighbor" % (counts[1])
                sys.exit()

    def apply_mask(self, config, mask):
        """ Use mask to prune edges in Delaunay triangulation that cross
            invalid regions.

        @param self The object pointer
        
        @param config Config object
        
        @param mask Mask object
        
        """
        # check for mask
        if not mask:
            return

        # set up new copies of triangles and neighbors
        masked_triangles = self._triangles.copy()
        masked_neighbors = self._neighbor_dict.copy()

        # loop over triangles
        for t in self._triangles:
            # loop over pairs of vertices
            for (ip1, ip2) in it.permutations(t, 2):
                
                # get coords of points
                x1 = config.ximages[ip1]
                y1 = config.yimages[ip1]
                x2 = config.ximages[ip2]
                y2 = config.yimages[ip2]

                # check if edge crosses mask boundary
                if not mask.is_valid_seg(x1, y1, x2, y2):

                    # if invalid, remove from masked_triangles and neighbors
                    # use discard (doesn't raise error if missing)
                    #    in case we've seen this pair before
                    masked_triangles.discard(t)
                    masked_neighbors[ip1].discard(ip2)
                    masked_neighbors[ip2].discard(ip1)

        # log and finish
        print "after masking, removed %d triangles" % (len(self._triangles) -
                                                       len(masked_triangles))
        self._neighbor_dict = masked_neighbors
        self._triangles = masked_triangles
         
    def area_of_point_set(self, ips, xc, yc):
        """ Compute an occupied area for a set of (mostly contiguous) points.

        This area is neither the convex hull nor the area of the Voronoi cells.
        It is the area of all interior triangles (3 vertices in set)
            plus the area of the trapezoids formed by 2 vertices
                and the midpoints of edges to the 3rd vertex
                    (2 vertices in set)
            plus the area of the small triangles formed by 1 vertex
                and the midpoints of edges to the 2nd and 3rd vertices
                    (1 vertex in set).

        @param self The object pointer
        
        @param ips List of particle ids in *images
        
        @param xc List of x coords with indices in ips
        
        @param yc List of y coords with indices in ips
        
        @retval area Float for area as described above
        
        @retval edges List of list of coordinates,
            edges[0] is x coords and edges[1] is y coords
            (easy for matplotlib.fill)
            
        """
        # set up storage
        area = 0
        edges = [[], []]

        # loop over all triangles
        for t in self._triangles:

            # find which vertices are in set
            is_in_set = [ip in ips for ip in t]
            # count Trues
            n_in_set = sum(is_in_set)

            # if no vertices in set, skip
            if n_in_set == 0:
                continue

            # if all in set, add area of whole triangle
            elif n_in_set == 3:
                area += self._triangle_area(xc[t[0]], yc[t[0]],
                                            xc[t[1]], yc[t[1]],
                                            xc[t[2]], yc[t[2]])
                
            # if one in set, add area of little triangle
            elif n_in_set == 1:
                # get vertex of point in triangle
                ip_in_set = is_in_set.index(True)
                x1, y1 = xc[t[ip_in_set]], yc[t[ip_in_set]]

                # get vertices of midpoints to other two vertices
                x2, y2 = (xc[t[(ip_in_set+1)%3]] + xc[t[ip_in_set]]) / 2.0,
                         (yc[t[(ip_in_set+1)%3]] + yc[t[ip_in_set]]) / 2.0
                x3, y3 = (xc[t[(ip_in_set+2)%3]] + xc[t[ip_in_set]]) / 2.0,
                         (yc[t[(ip_in_set+2)%3]] + yc[t[ip_in_set]]) / 2.0

                # add to area and edges
                area += self.triangleArea(x1, y1, x2, y2, x3, y3)
                edges[0].extend([x2, x3])
                edges[1].extend([y2, y3])

            # if two in set, add area of trapezoid
            elif n_in_set == 2:
                # get vertices of points in triangle
                ip_out_set = is_in_set.index(False)
                x1, y1 = xc[t[(ip_out_set+1)%3]], yc[t[(ip_out_set+1)%3]]
                x2, y2 = xc[t[(ip_out_set+2)%3]], yc[t[(ip_out_set+2)%3]]

                # get vertices of midpoints to outside vertex
                x3, y3 = (xc[t[(ip_out_set+2)%3]] + xc[t[ip_out_set]]) / 2.0,
                         (yc[t[(ip_out_set+2)%3]] + yc[t[ip_out_set]]) / 2.0
                x4, y4 = (xc[t[(ip_out_set+1)%3]] + xc[t[ip_out_set]]) / 2.0,
                         (yc[t[(ip_out_set+1)%3]] + yc[t[ip_out_set]]) / 2.0

                # add to area and edges
                area += self._trapezoid_area(x1, y1, x2, y2, x3, y3, x4, y4)
                edges[0].extend([x3, x4])
                edges[1].extend([y3, y4])

            # if n_in_set is anything else, error
            else:
                print "ERROR: n in set is", n_in_set

        # return
        return area, edges

    def _triangle_area(self, x1, y1, x2, y2, x3, y3):
        # http://mathworld.wolfram.com/PolygonArea.html
        return abs(0.5 * (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y1 - x1*y3))

    def _trapezoid_area(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # http://mathworld.wolfram.com/PolygonArea.html
        # vertices must be ordered contiguously (CW or CCW)
        return abs(0.5 * (x1*y2 - x2*y1 + x2*y3 - x3*y2 + x3*y4 - x4*y3 +
                          x4*y1 - x1*y4))
