"""
@package bravais
@author Anna Schneider
@version 0.1
@brief Contains class for BravaisLattice
"""

# import from standard library
import itertools

# import external packages
import numpy as np
from scipy import optimize, spatial

# import modules in this package
from config import Coord

class BravaisLattice:
    """ Class for matching Bravais lattices to sets of 2d points. """    
    
    def __init__(self):
        """ Constructor. """
        pass
    
    def _error(self, xy_tuple, coord_pairs, rcut_sq, kl_pairs):
        """ Wrapped by BravaisLattice.error;
            This gets called many times per call to BravaisLattice.fit
            
        @param xy_tuple (x1, y1, x2, y2)
        
        @param coord_pairs Nx2 ndarray of x,y coords

        @param rcut_sq Number for squared distance cutoff

        @param kl_pairs Ndarray of k,l pairs, eg from BravaisLattice.kl_pairs()
                    
        @retval error
            
        """
        # set up target Bravais lattice
        kx1 = kl_pairs[:,0] * xy_tuple[0]
        lx2 = kl_pairs[:,1] * xy_tuple[2]
        ky1 = kl_pairs[:,0] * xy_tuple[1]
        ly2 = kl_pairs[:,1] * xy_tuple[3]
        bravais_pairs = np.vstack((kx1 + lx2, ky1 + ly2)).transpose()
        
        # get squared distance between every Bravais point and every coord point
        # sq_dists has shape (n_bravais_pairs, n_coord_pairs)
        sq_dists = spatial.distance.cdist(bravais_pairs, coord_pairs,
                                          'sqeuclidean')
        # get min dist for each coord
        min_sq_dists = np.min(sq_dists, axis=0)
       
        # apply error function
        scaled_sq_dists = min_sq_dists / rcut_sq
        errors = np.where(scaled_sq_dists < 1.0, scaled_sq_dists, 1.0)
        error = np.mean(errors)
        
     #   error = 0
     #   for coord in coords:
            # find closest Bravais point to each actual particle
     #       closest_dist_sq = min([(coord.x-bp.x)**2 + (coord.y-bp.y)**2 for bp in bravais])
            # piecewise error function
     #       error += min(closest_dist_sq / rcut_sq, 1.0)
     #   error /= len(coords)
    #    error = sum([min(min([(coord.x-bp.x)**2 + (coord.y-bp.y)**2 for bp in bravais]) / rcut_sq, 1.0)]) / len(coords)
        
        return error
                
    def expand(self, x1, y1, x2, y2, n_kls=2):
        """ Expand a local Bravais lattice around on vectors (x1,y1) and
            (x2,y2), with lattice constants k,l in [-n_kls, n_kls].
            Returns a list of Coords.
                    
        """
        # set up target Bravais lattice up to +/- max_const
        bravais = []
        for k,l in itertools.product(xrange(-n_kls, n_kls+1), repeat=2):
            bravais.append(Coord(k*x1 + l*x2, k*y1 + l*y2))
        
        # return
        return bravais
        
    def kl_pairs(self, n_kls):
        """ Utility function to return a nx2 array for all combinations of
                lattice constants k,l in [-n_kls, n_kls].
            Generally not called directly.
        """
        kl_grid = np.indices((n_kls*2+1, n_kls*2+1))-n_kls
        return kl_grid.reshape(2, (n_kls*2+1)**2).transpose()
        
    def error(self, xy_tuple, coords, n_kls=2, r_cut=1):
        """ Get the error of the Bravais lattice described by the (x,y)
            points in xytuple to the provided coords using the error function
            sum_{coords} [ min[ min_{bravais}[ f(coord, bravais) ], 1.0 ] ]
            for f(coord, bravais) = dist(coord, bravais)**2/(rcut_sq)

        @param xy_tuple (x1, y1, x2, y2)

        @param coords List of Coord objects, with the origin at the reference
            position of the Bravais lattice        
           
        @param n_kls Int such that lattice constants k,l are 
            in [-n_kls, n_kls]
            
        @param r_cut Number for cutoff distance for error function
        
        @retval error
            
        """
        kl_pairs = self.kl_pairs(n_kls)
        coord_pairs = np.array([[c.x, c.y] for c in coords])
        return self._error(xy_tuple, coord_pairs, r_cut**2, kl_pairs)        
                
    def fit(self, coords, n_kls=2, r_cut=1, max_dist=np.inf, min_dist=0):
        """ Finds the 2d Bravais lattice that best fits a set of coords.
        
        @param self The object pointer
        
        @param coords List of Coord objects, with the origin at the reference
            position of the Bravais lattice
            
        @param n_kls Int such that lattice constants k,l are 
            in [-n_kls, n_kls]
            
        @param r_cut Number for cutoff distance for error function
            
        @param max_dist Number for maximum length of b vector
        
        @param min_dist Number for minimum length of a vector
                    
        @retval List of Coord objects for points in Bravais lattice
        
        @retval error Number for error in lattice fit
        
        """
        # set up parameters for optimizing Bravais lattice
        best_error = np.Inf
        best_bravais = None
        self._max_dist_sq = max_dist**2
        self._min_dist_sq = min_dist**2
        kl_pairs = self.kl_pairs(n_kls)
        
        # sort Coords by theta
        coords.sort(cmp = lambda u,v: cmp(u.theta, v.theta))
        # store x,y from sorted coords 
        coord_pairs = np.array([[c.x, c.y] for c in coords])
        
       # loop over particle pairs with a clockwise of b
        for ia in xrange(len(coords)):
            if best_error is not np.Inf:
                break
            for ib in xrange(ia+1 - len(coords), ia):
                # set up args
                init_xy_tuple = (coords[ia].x, coords[ia].y,
                                 coords[ib].x, coords[ib].y)
                args = (coord_pairs, r_cut**2, kl_pairs)

                # check initial guess, may be done already!
                unopt_error = self._error(init_xy_tuple, *args)
                if unopt_error < 1e-6:
                    best_error = unopt_error
                    best_bravais = self.expand(*init_xy_tuple, n_kls=n_kls)
                    break
                    
                # if not, optimize
                opt_xy_tuple = optimize.fmin_cobyla(self._error,
                                                    init_xy_tuple,
                                                    [self._constr_max1,
                                                     self._constr_max2,
                                                     self._constr_min1,
                                                     self._constr_min2],
                                                    args=args,
                                                    consargs=(),
                                                    disp=0)
                opt_error = self._error(opt_xy_tuple, *args)

                # check for best fit lattice
                if opt_error < best_error - 1e6:
                    best_error = opt_error
                    best_bravais = self.expand(*opt_xy_tuple, n_kls=n_kls)                    
                    break
                
        return best_bravais, best_error
        
    def _constr_max1(self, x):
        # first vec mag < max_dist
        return self._max_dist_sq - x[0]*x[0] + x[1]*x[1]

    def _constr_max2(self, x):
         # second vec mag < max_dist
        return self._max_dist_sq - x[2]*x[2] + x[3]*x[3]
        
    def _constr_min1(self, x):
        # first vec mag > min_dist
        return x[0]*x[0] + x[1]*x[1] - self._min_dist_sq
        
    def _constr_min2(self, x):
         # first vec mag > min_dist
        return x[2]*x[2] + x[3]*x[3] - self._min_dist_sq
