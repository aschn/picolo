"""
@package shapes
@author Anna Schneider
@version 0.1
@brief Contains classes for Shape, FourierShape, ZernikeShape, UnitCellShape,
    and factory methods shape_factory_from_values and shape_factory_from_coords
"""

# import from standard library
import math
import copy
import itertools

# import external packages
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class Shape:
    """ Base class for shape descriptors.
        Stores an ordered list of named variables that specify the 
            components of the shape descriptor vector,
            and the value associated with each variable.
        Implement build_from_coords() and _postprocessing()
            to provide shape-specific functionality.
    
    """
    def __init__(self, var_names=[], vals=[], **kwargs):  
        """ Constructor
        
        @param var_names List of variable names
        
        @param vals List of numbers for the value of each variable
        
        @param **kwargs Optional arguments, to be set as attributes
        
        """
        # test
        if len(var_names) != len(vals):
            msg = 'Variables and vals must have same length. '
            msg += 'Got lengths %d and %d' % (len(var_names), len(vals))
            raise ValueError(msg)
                    
        # set up variables and vals
        self._var_names = []
        self._vals = np.asarray([])
        for iv in range(len(vals)):
            self.put_component(var_names[iv], vals[iv])
        
        # store any keyword-args
        if kwargs:
            self._params = kwargs 
        else:
            self._params = dict()
            
        # clean up
        self._postprocessing()
                        
    def _postprocessing(self):
        """ Implement this method to provide class-specific functionality
            after construction. """
        pass
                        
    def __len__(self):
        return len(self._vals)

    def get(self, var_name):
        """ Returns named variable or parameter called var_name.
            If variable not found, returns None. """
        if var_name in self._var_names:
            iv = self._var_names.index(var_name)
            return self._vals[iv]
        elif var_name in self._params:
            return self._params[var_name]
        else:
            return None
                                
    def get_vals(self, norm=False):
        """ Returns ndarray of values, normalized only if requested. """
        if norm:
            return self._vals / self.mag()
        else:
            return self._vals
            
    def iter_components(self):
        """ Iterate over (var_name, val) pairs of feature vector components. """
        for iv in range(len(self._var_names)):
            yield self._var_names[iv], self._vals[iv]
            
    def copy(self):
        """ Deep copy of self. """
        return copy.deepcopy(self)
        
    def mag(self):
        """ Get magnitude of vector of vals. """
        return np.linalg.norm(self._vals)
                
    def subset(self, new_var_names):
        """ Returns a copy of the shape with some components removed.
        
        @param new_var_names List of variable names to keep
        
        @retval Shape object of the same type as self
        
        """
        # make copy of self
        new_shape = self.copy()
        
        # check that we have all ls
        extra_vars = set(new_shape._var_names) - set(new_var_names)
        if len(extra_vars) > 0:
            msg = 'New variables must be a subset of existing variables. '
            msg += 'Got extra variables %s' % ', '.join(list(extra_vars))
            raise ValueError(msg)

        # drop unneeded vals
        for name in new_shape._var_names:
            if name not in new_var_names:
                new_shape.drop_component(name)
        
        # return
        return new_shape
        
    def has_component(self, var_name):
        """ Returns True if there is a variable that matches var_name,
            False otherwise.
            
        """
        if var_name in self._var_names:
            return True
        else:
            return False
            
    def drop_component(self, var_name):
        """ Removes variable called var_name if it is present;
            does nothing otherwise.
            
        """
        if self.has_component(var_name):
            iv = self._var_names.index(var_name)
            del self.var_name[iv]
            self._vals = np.delete(self._vals, self._vals[iv])
            
    def put_component(self, var_name, val):
        """ Sets variable called var_name with value val;
            overwrites value if the variable already exists.
            
        """
        if self.has_component(var_name):
            iv = self._var_names.index(var_name)
            self._vals[iv] = val
        else:
            self._var_names.append(var_name)
            self._vals.append(val)
            
    def put_param(self, attr_name, val):
        """ Sets parameter attribute with value val;
            overwrites value if the parameter already exists.
            
        """
        self._params[attr_name] = val
            
    def build_from_coords(self, neighbor_coords):
        """ Implement this method to provide class-specific functionality
            for calculating the shape descriptor variables from a set of
            points.
        
        """
        pass

class FourierShape(Shape):
    """ Shape object where component variables are rotation-invariant
        Fourier descriptors.
    
    """
    def _postprocessing(self):
        # convert variable names to ints if needed
        # assuming they are FourierShape-like l variable names
        for iv in range(len(self._var_names)):
            self._var_names[iv] = int(self._var_names[iv])
    
    def build_from_coords(self, neighbor_coords):    
        """ Update with rotation invariant Fourier descriptors of neighboring
            points about reference position, indexed by ints l.

        @param self The object pointer
        
        @param neighbor_coords List of Coord objects for neighboring points,
            with the origin at the reference position
                
        """
        # set up storage
        self.put_param('n_neighbors', len(neighbor_coords))
        complexvals = np.zeros(max(var_names), dtype=complex)
        
        # loop over neighbors
        for coord in neighbor_coords:
            # add component to running sum
            for iv in self._var_names:
                complexvals.real[iv] += math.cos(iv*coord.theta)
                complexvals.imag[iv] += math.sin(iv*coord.theta)
        
        # normalize sums
        complexvals /= float(max(self.get('n_neighbors'), 1))
        
        # set vals
        for iv in self._var_names:
            val = np.abs(complexvals[iv])
            self.put_component(iv, val)

class ZernikeShape(Shape):
    """ Shape object where component variables are rotation-invariant
        Zernike moments, indexed by tuples of ints (n,m).
    
    """
    def _postprocessing(self):
        # convert variable names to tuples (n,m) if needed
        # assuming they are FourierShape-like l variable names
        if not isinstance(self._var_names[0], tuple):
            raise RuntimeWarning('dropping all values from ZernikeShape')
            self._var_names = []
            self._vals = np.asarray([])
            nms = self._ls2nms(self._var_names)
            for nm in nms:
                self.put_component(nm, 0)
               
    def build_from_coords(self, neighbor_coords):    
        """ Update with Zernike rotation invariant moments corresponding to
            positions of neighboring points about reference position.
            
        @param self The object pointer
        
        @param neighbor_coords List of Coord objects for neighboring points,
            with the origin at the reference position
                            
        """ 
        # notation follows "Invariant Image Recognition by Zernike Moments"
        # by Khotanzad and Hong, 1990 IEEE.

        # set up storage
        self.put_param('n_neighbors', len(neighbor_coords))
        var_names = ref_shape._var_names
        complexvals = np.zeros(len(var_names), dtype=complex)
        ns = [self._nm2n(nm) for nm in var_names]
        ms = [self._nm2m(nm) for nm in var_names]

        # loop over neighbors
        for coord in neighbor_coords:
            # add component to running sum
            rscaled = coord.r / self.get('neighbor_dist')
            for iv in range(len(var_names)):
                rnm = self._rnm(ns[iv], ms[iv], rscaled)
                coeff = rnm * (ns[iv]+1.0)/math.pi
                complexvals.real[iv] += math.cos(ms[iv]*coord.theta) * coeff
                complexvals.imag[iv] += math.sin(ms[iv]*coord.theta) * coeff
        
        # normalize sum
        complexvals /= float(max(self.get('n_neighbors'), 1))

        # set vals
        for iv in range(len(var_names)):
            val = np.abs(complexvals[iv])
            self.put_component(var_names[iv], val)
        
    def _ls2nms(self, ls):
        """ Convert list of Fourier indices to (n,m) Zernike index pairs
        
        @param self The object pointer
        
        @param ls List for Fourier index
        
        @retval nms List of 2-tuples of ints
        
        """
        nms = []
        for l in ls:
            if l % 2 == 1:
                # max(m) < l
                nms += [(l, m) for m in range(0, l, 2)]
            else:
                # max(m) == l
                nms += [(l, m) for m in range(0, l+1, 2)]
        return nms
        
    def _nm2n(self, nm):        
        return nm[0]

    def _nm2m(self, nm):
        return nm[1]
        
    def _rnm(self, n, m, r):
        """ Compute radial polynomial part of Zernike polynomial """
        r_sum = 0
        m = int(abs(m))
        u = int((n-m)/2)
        v = int((n+m)/2)
        for s in range(0, u+1):
            numerator = pow(-1, s) * math.factorial(int(n-s)) * pow(r, n-2*s)
            try:
                denominator = math.factorial(s) * math.factorial(v-s) * math.factorial(u-s)
            except ValueError:
                raise ValueError('(s,n,m,u,v) = (%d,%d,%d,%d,%d)' % (s, n, m, u, v))
            r_sum += numerator / denominator
        return r_sum
        

class UnitCellShape(Shape):
    """ Shape object where component variables are 2d unit cell parameters
            (a, b, angle).
        a and b are the lengths of the unit cell vectors, where a < b.
        If (a,0) is the polar coordinates of vector a, then (b, angle) is
            the polar coordinates of vector b, such that 0 < angle < 2pi.
        get('theta') returns the angle in radians;
        get('degrees') returns the angle in degrees.
    
    """
    def _postprocessing(self):
        if self.get('a') and self.get('b') and self.get('theta'):
            self.put_param('is_cell', True)
        else:
            self.put_param('is_cell', False)

        # if not set, use hard-coded default params for computing unit cells
        defaults = {'min_dist': 14.0, 'max_dist': self.get('neighbor_dist'),
                    'min_error': 0.2, 'r_cut': 7.0,
                    'target_angle': math.radians(75.0),
                    'max_angle': math.radians(120.0),
                    'min_angle': math.radians(50.0)
                    }
        for k, v in defaults.iteritems():
            if not self.get(k):
                self.put_param(k, v)

    def build_from_coords(self, neighbor_coords, do_plot=False):    
        """ Update with unit cell (a,b,theta) corresponding to
            positions of neighboring points about reference position.
            
        @param self The object pointer
        
        @param neighbor_coords List of Coord objects for neighboring points,
            with the origin at the reference position
            
        @param do_plot Bool for whether or not to show a diagnostic plot
            interactively
                
        """ 
        # set up storage
        coords_in_range = []
                
        # loop over particles to get coordinates
        for coord in neighbor_coords:
            if coord.r > self.get('min_dist'):
                coords_in_range.append(coord)
                        
        # can't have unit cell if fewer than 4 neighbors
        if len(coords) < 4:
            self.invalidate()
            return
            
        # find best optimized Bravais lattice
        bravais, error = self._best_bravais_lattice(coords_in_range)            
        
        # decide if good unit cell
        if error > self.min_error:
            self.invalidate()
            return

        # find good unit cell using Bravais lattice
        a, b, degrees = self._bravais_to_unit_cell(bravais)

        if not a:
            coord_string = "\n".join([repr(coord) for coord in coords])
            raise RuntimeWarning('no unit cell for coords: %s' % coord_string)
            self.invalidate()
            return
                
        # plot
        if do_plot:
            plt.scatter([cp.x for cp in coords],
                        [cp.y for cp in coords], c='r', s=70)
            plt.title(repr(error))
            if bravais:
                plt.scatter([bp.x for bp in bravais],
                            [bp.y for bp in bravais], c='b')
            plt.show()   
            
        # return
        self.put_param('is_cell', True)
        self.put_component('a', a)
        self.put_component('b', b)
        self.put_component('theta', math.radians(degrees))
        self.put_param('degrees', degrees)
                    
    def _lattice_error(self, xy_tuple, coords, rcut):
        """ Get the error of the Bravais lattice described by the (x,y)
            points xytuple[0:2] and xytuple[2:4] to the provided coords
            using the error function
            sum_{coords} [ min[ min_{bravais}[ f(coord, bravais) ], 1.0 ] ]
            for f(coord, bravais) = dist(coord, bravais)**2/(rcut)**2
            
        """
        x1, y1, x2, y2 = xy_tuple

        # set up target Bravais lattice
        bravais = self._bravais_lattice(x1, y1, x2, y2)
        
        # find closest Bravais point to each actual particle
        # error is piecewise linear function in distance
        error = 0
        rcut_sq = rcut*rcut
        for ic, coord in enumerate(coords):
            closest_dist_sq = np.Inf
            for bp in bravais:
                dist_sq = (coord.x-bp.x)*(coord.x-bp.x) +
                            (coord.y-bp.y)*(coord.y-bp.y)
                if dist_sq < closest_dist_sq:
                    closest_dist_sq = dist_sq
            if closest_dist_sq < rcut_sq:
                error += closest_dist_sq / rcut_sq
            else:
                error += 1.0
        error /= float(len(coords))
        
        return error
        
    def _bravais_lattice(self, x1, y1, x2, y2, max_const=2):
        """ Set up a local Bravais lattice based on vectors (x1,y1) and
            (x2,y2), with lattice constants k,l in [-max_const, max_const].
            Returns a list of Coords.
                    
        """
        bravais_kls.sort(key = lambda u: sum([x*x for x in u]))
        
        # set up target Bravais lattice up to +/- max_const
        lattice_consts = range(-max_const, max_const+1)
        bravais = []
        for k,l in itertools.product(lattice_consts, repeat=2):
            bravais.append( Coord(k*x1 + l*x2, k*y1 + l*y2) )
        
        # sort so smallest radial distances are first
        bravais.sort(key = lambda c: c.r)
        
        # return
        return bravais
        
    def _best_bravais_lattice(self, coords):
        """ Finds the 2d Bravais lattice that best matches a set of coords.
        
        @param self The object pointer
        
        @param coords List of Coord objects, with the origin at the reference
            position of the Bravais lattice
            
        @retval List of Coord objects for points in Bravais lattice
        
        """
        # if ok, sort tuples by theta
        coords.sort(cmp = lambda u,v: cmp(u.theta, v.theta))
        
        # find best optimized Bravais lattice
        best_error = np.Inf
        best_bravais = None
        max_dist_sq = self.get('max_dist')**2
        min_dist_sq = self.get('min_dist')**2
        constr_max1 = lambda x: max_dist_sq - x[0]*x[0] + x[1]*x[1]  # first vec mag < max_dist
        constr_max2 = lambda x: max_dist_sq - x[2]*x[2] + x[3]*x[3]  # second vec mag < max_dist
        constr_min1 = lambda x: x[0]*x[0] + x[1]*x[1] - min_dist_sq  # first vec mag > min_dist
        constr_min2 = lambda x: x[2]*x[2] + x[3]*x[3] - min_dist_sq  # first vec mag > min_dist
        
       # loop over particle pairs with a clockwise of b
        for ia in range(0, len(coords)):
            if best_error is not np.Inf:
                break
            for ib in range(ia+1 - len(coords), ia):
                init_xy_tuple = (coords[ia].x, coords[ia].y,
                                 coords[ib].x, coords[ib].y)
                args = (coords, self.r_cut)
                opt_xy_tuple = optimize.fmin_cobyla(self._lattice_error,
                                                    init_xy_tuple,
                                                    [constr_max1, constr_max2,
                                                     constr_min1, constr_min2],
                                                    args = args,
                                                    consargs = (),
                                                    disp=0)
                error = self._lattice_error(opt_xy_tuple, *args)

                # check for best fit lattice
                if error < best_error - 1e6:
                    best_error = error
                    best_bravais = self._bravais_lattice(*opt_xy_tuple)
                    break
                
        return best_bravais, best_error
        
    def _bravais_to_unit_cell(self, bravais):
        """ Find the unit cell parameters (a,b,angle) that describe the
            input Bravais lattice and conform to the target parameter ranges.
            
        @param bravais List of Coord objects for points in Bravais lattice
        
        @retval tuple of numbers (a,b,angle) with angle in degrees

        """                
        best_a = None
        best_b = None
        best_angle = None
        best_area = None

        # loop through pairs
        for ia in range(1, len(best_bravais)):
            for ib in range(1, len(best_bravais)):
                if ia is not ib:
                    # enforce a<=b
                    a = best_bravais[ia].r
                    b = best_bravais[ib].r
                    if ( (a > b) or (a < self.get('min_dist')) or
                        (b > self.get('max_dist')) ):
                        continue
                    
                    # get angle, in range (0,2*pi)
                    angle_diff = best_bravais[ib].theta - best_bravais[ia].theta
                    angle = np.remainder(angle_diff, 2.0*math.pi)
                    area = self.area(a, b, angle)
                    
                    # enforce 45<=angle<120
                    if angle < self.min_angle or angle > self.max_angle:
                        continue
                    elif not best_angle:
                        best_a = a
                        best_b = b
                        best_angle = angle
                        best_area = area
                    else:
                        conditions = (abs(self.get'(target_angle')-angle)+1e-4 <
                                        abs(self.get('target_angle')-best_angle)) and
                                     (area < 1.1 * best_area)
                        if conditions:
                            best_a = a
                            best_b = b
                            best_angle = angle
                            best_area = area
                        
        return best_a, best_b, best_angle
                    
    def area(self, a=None, b=None, theta=None):
        """ With no arguments, calculate the area of this unit cell;
            returns 0 if invalid.
            With arguments, calculates the area of that unit cell,
            assuming theta in radians.
            
        """
        if a and b and theta:
            return float(a) * float(b) * math.sin(theta)
        elif self.get('is_cell'):
            return float(self.get('a')) * float(self.get('b')) * 
                    math.sin(self.get('theta'))
        else:
            return 0
            
    def invalidate(self):
        """ Mark this unit cell as invalid. """
        self.set_param('is_cell', False)
        for iv in range(len(self._var_names)):
            self._vals[iv] = None
        
def shape_factory_from_values(shape_type, variables, vals, optdata=dict()):
    """ Factory function to create a shape given variable names and values.
        Valid types must contain the substrings 'UnitCell', 'Fourier',
        or 'Zernike' (case insensitive).
    
    @param shape_type String specifying a valid shape type
    
    @param variables List of variable names
    
    @param vals List of numbers of variable values
    
    @param optdata Dict of additional attribute name-val pairs
        
    @retval Shape object of appropriate type
    
    """
    lower_case_str = shape_type.lower()

    if 'unitcell' in lower_case_str:
        return UnitCellShape(variables, vals, optdata)
    elif 'fourier' in lower_case_str:
        return FourierShape(variables, vals, optdata)
    elif 'zernike' in lower_case_str:
        return ZernikeShape(variables, vals, optdata)
    else:
        raise ValueError('invalid shape type %s' % shape_type)
        
def shape_factory_from_coords(neighbor_coords, ref_shape):
    """ Factory function to create a shape given coordinates.
        Use a reference shape as a template.
        
    @param neighbor_coords List of Coord objects for neighboring points
        
    @param ref_shape Shape object of appropriate type for template
        
    @retval Shape object of appropriate type
    
    """
    if isinstance(ref_shape, UnitCellShape) or
            isinstance(ref_shape, FourierShape) or
            isinstance(ref_shape, ZernikeShape):
        shape = ref_shape.copy()
    else:
        raise ValueError('invalid shape type %s' % type(ref_shape))
        
    shape.build_from_coords(neighbor_coords)
    return shape