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
import warnings
import logging

# import external packages
import numpy as np
import matplotlib.pyplot as plt

# import modules in this package
from config import BravaisLattice

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
        # (in)validate
        if len(self._var_names) == 0:
            self.invalidate()
        else:
            self.put_param('is_valid', True)
            
        # set type
        self.put_param('type', 'Generic')
                        
    def __len__(self):
        return len(self._vals)

    def get(self, var_name):
        """ Returns named variable or parameter called var_name.
            If variable not found, raises KeyError. """
        if var_name in self._var_names:
            iv = self._var_names.index(var_name)
            return self._vals[iv]
        elif var_name in self._params:
            return self._params[var_name]
        else:
            raise KeyError("Nothing found for %s in vars (%s) or params (%s)" % (str(var_name),
                                                                             ', '.join(self._var_names),
                                                                             ', '.join(self._params.keys())))
                                
    def get_vals(self, norm=False):
        """ Returns ndarray of values, normalized only if requested. """
        if norm:
            return self._vals / self.mag()
        else:
            return self._vals

    def get_components(self, norm=False):
        """ Returns list of all variable names. """
        return self._var_names
            
    def iter_components(self):
        """ Iterate over (var_name, val) pairs of feature vector components. """
        for iv in range(len(self._var_names)):
            yield self._var_names[iv], self._vals[iv]

    def iter_params(self):
        """ Iterate over (var_name, val) pairs of non-feature vector parameters. """
        for var, val in self._params.iteritems():
            yield var, val
            
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
        
        # check that new names are a subset of old names
        if not new_var_names <= new_shape._var_names:
            extra_vars = set(new_var_names) - set(new_shape._var_names)
            extra_var_strings = [str(var) for var in extra_vars]
            msg = 'New variables must be a subset of existing variables. '
            msg += 'Got extra variables %s' % ', '.join(extra_var_strings)
            raise ValueError(msg)

        # drop unneeded vals
        for name in self._var_names:
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
            del self._var_names[iv]
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
            self._vals = np.append(self._vals, val)
            
    def put_param(self, attr_name, val):
        """ Sets parameter attribute with value val;
            overwrites value if the parameter already exists.
            
        """
        self._params[attr_name] = val
        
    def invalidate(self):
        """ Mark this shape as invalid. """
        self.put_param('is_valid', False)
        self._vals = np.empty_like(self._vals)
            
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
        # if no var names set, set some by default
        if len(self._var_names) == 0:
            self._var_names = range(2, 25, 2)
            self._vals = np.zeros(len(self._var_names))
            self.invalidate()
        else:
            self.put_param('is_valid', True)
            
        # convert variable names to ints if needed
        # assuming they are FourierShape-like l variable names
        for iv in range(len(self._var_names)):
            self._var_names[iv] = int(self._var_names[iv])
            
        # set type
        self.put_param('type', 'Fourier')
    
    def build_from_coords(self, neighbor_coords):    
        """ Update with rotation invariant Fourier descriptors of neighboring
            points about reference position, indexed by ints l.

        @param self The object pointer
        
        @param neighbor_coords List of Coord objects for neighboring points,
            with the origin at the reference position
                
        """
        # set up storage
        self.put_param('n_neighbors', len(neighbor_coords))
        complexvals = np.zeros(len(self._var_names), dtype=complex)
        
        # loop over neighbors
        for coord in neighbor_coords:
            # add component to running sum
            for iv in self._var_names:
                index = self._var_names.index(iv)
                complexvals.real[index] += math.cos(iv*coord.theta)
                complexvals.imag[index] += math.sin(iv*coord.theta)
        
        # normalize sums
        complexvals /= float(max(self.get('n_neighbors'), 1))
        
        # set vals
        for index, iv in enumerate(self._var_names):
            val = np.abs(complexvals[index])
            self.put_component(iv, val)
            
        # validate
        self.put_param('is_valid', True)

class ZernikeShape(Shape):
    """ Shape object where component variables are rotation-invariant
        Zernike moments, indexed by tuples of ints (n,m).
        
        Required parameters to set with put_param:
            
            neighbor_dist = radius of the disc on which radial Zernike
                polynomials are defined
    
    """
    def _postprocessing(self):
        # if no var names set, set some by default
        if len(self._var_names) == 0:
            self._var_names = range(2, 25)
            self.invalidate()
        else:
            self.put_param('is_valid', True)

        # convert variable names to tuples (n,m) if needed
        # assuming they are FourierShape-like l variable names
        if len(self._var_names) > 0:
            if not isinstance(self._var_names[0], tuple):
                warnings.warn('dropping all values from ZernikeShape',
                              RuntimeWarning)
                nms = self._ls2nms(self._var_names)
                self._var_names = []
                self._vals = np.asarray([])
                for nm in nms:
                    self.put_component(nm, 0)
        # set type
        self.put_param('type', 'Zernike')
               
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
        complexvals = np.zeros(len(self._var_names), dtype=complex)
        ns = [self._nm2n(nm) for nm in self._var_names]
        ms = [self._nm2m(nm) for nm in self._var_names]

        # loop over neighbors
        for coord in neighbor_coords:
            # add component to running sum
            rscaled = coord.r / self.get('neighbor_dist')
            for iv in range(len(self._var_names)):
                rnm = self._rnm(ns[iv], ms[iv], rscaled)
                coeff = rnm * (ns[iv]+1.0)/math.pi
                complexvals.real[iv] += math.cos(ms[iv]*coord.theta) * coeff
                complexvals.imag[iv] += math.sin(ms[iv]*coord.theta) * coeff
        
        # normalize sum
        complexvals /= float(max(self.get('n_neighbors'), 1))

        # set vals
        for iv in range(len(self._var_names)):
            val = np.abs(complexvals[iv])
            self.put_component(self._var_names[iv], val)
        
        # validate
        self.put_param('is_valid', True)
        
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
    
        Required parameters, set by default:
            
            neighbor_dist = radius of the disc on which radial Zernike
                polynomials are defined
    
    """
    def _postprocessing(self):
        try:
            if self.get('a') and self.get('b') and self.get('degrees'):
                self.put_param('is_valid', True)
            elif self.get('a') and self.get('b') and self.get('theta'):
                self.put_param('is_valid', True)
            else:
                self.put_param('is_valid', False)
        except KeyError:
            self.put_param('is_valid', False)

        # if not set, use hard-coded default params for computing unit cells
        defaults = {'min_dist': 14.0, 'max_dist': self.get('neighbor_dist'),
                    'min_error': 0.2, 'r_cut': 7.0,
                    'target_angle': math.radians(75.0),
                    'max_angle': math.radians(120.0),
                    'min_angle': math.radians(50.0)
                    }
        for k, v in defaults.iteritems():
            try:
                self.get(k)
            except KeyError:
                self.put_param(k, v)
                
        # set angle in degrees and radians
        if self.get('is_valid'):
            try:
                self.get('theta')
            except KeyError:
                self.put_param('theta', math.radians(self.get('degrees')))
            try:
                self.get('degrees')
            except KeyError:
                self.put_component('degrees', math.degrees(self.get('theta')))
            
        # set type
        self.put_param('type', 'UnitCell')

    def build_from_coords(self, neighbor_coords):
        self.build_from_coords(neighbor_coords, False)
        
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
        if len(coords_in_range) < 4:
            self.invalidate()
            return
            
        # find best optimized Bravais lattice
        bl = BravaisLattice()
        bravais, error = bl.fit(coords_in_range,
                                r_cut=self.get('r_cut'),
                                min_dist=self.get('min_dist'),
                                max_dist=self.get('max_dist'))

        
        # decide if good unit cell
        if error > self.get('min_error'):
            self.invalidate()
            return

        # find good unit cell using Bravais lattice
        try:
            a, b, degrees = self._bravais_to_unit_cell(bravais)
        except:
            coord_string = "\n\t".join([repr(coord) for coord in coords_in_range])
            raise RuntimeWarning('no unit cell for coords:\n\t%s' % coord_string)
            self.invalidate()
            return
                
        # plot
        if do_plot:
            plt.scatter([cp.x for cp in coords_in_range],
                        [cp.y for cp in coords_in_range], c='r', s=70)
            plt.title(repr(error))
            if bravais:
                plt.scatter([bp.x for bp in bravais],
                            [bp.y for bp in bravais], c='b')
            plt.show()   
            
        # return
        self.put_param('is_valid', True)
        self.put_component('a', a)
        self.put_component('b', b)
        self.put_component('degrees', degrees)
        self.put_param('theta', math.radians(degrees))
        
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
        bravais.sort(cmp = lambda u,v: cmp(u.r, v.r))

        # loop through pairs
        for ia in range(1, len(bravais)):
            for ib in range(1, len(bravais)):
                if ia is not ib:
                    # enforce a<=b
                    a = bravais[ia].r
                    b = bravais[ib].r
                    if ( (a > b) or (a < self.get('min_dist')) or
                        (b > self.get('max_dist')) ):
                        continue
                    
                    # get angle, in range (0,2*pi)
                    angle_diff = bravais[ib].theta - bravais[ia].theta
                    angle = np.remainder(angle_diff, 2.0*math.pi)
                    area = self.area(a, b, angle)
                    
                    # enforce 45<=angle<120
                    if angle < self.get('min_angle') or angle > self.get('max_angle'):
                        continue
                    elif not best_angle:
                        best_a = a
                        best_b = b
                        best_angle = angle
                        best_area = area
                    else:
                        diff_to_angle = abs(self.get('target_angle') - angle)
                        diff_to_best = abs(self.get('target_angle') - best_angle)
                        angle_condition = (diff_to_angle+1e-4 < diff_to_best)
                        area_condition = (area < 1.1 * best_area)
                        if angle_condition and area_condition:
                            best_a = a
                            best_b = b
                            best_angle = angle
                            best_area = area
                        
        logging.debug("a=%s, b=%s, theta=%s" % (best_a, best_b, best_angle))
        try:
            return best_a, best_b, math.degrees(best_angle)
        except TypeError:
            logging.debug("No unit cell found for Bravais %s, %s" % (bravais[1],
                                                                        bravais[2]))
            raise ValueError("No unit cell found for Bravais %s, %s" % (bravais[1],
                                                                        bravais[2]))
                    
    def area(self, a=None, b=None, theta=None):
        """ With no arguments, calculate the area of this unit cell;
            returns 0 if invalid.
            With arguments, calculates the area of that unit cell,
            assuming theta in radians.
            
        """
        if a and b and theta:
            return float(a) * float(b) * math.sin(theta)
        elif self.get('is_valid'):
            return float(self.get('a')) * float(self.get('b')) * math.sin(self.get('theta'))
        else:
            return 0.0
                    
def shape_factory_from_values(shape_type, variables, vals, optdata=dict()):
    """ Factory function to create a shape given variable names and values.
        Valid types must contain the substrings 'UnitCell', 'Fourier',
        'Zernike', or 'Generic' (case insensitive).
    
    @param shape_type String specifying a valid shape type
    
    @param variables List of variable names
    
    @param vals List of numbers of variable values
    
    @param optdata Dict of additional attribute name-val pairs
        
    @retval Shape object of appropriate type
    
    """
    lower_case_str = shape_type.lower()

    if 'generic' in lower_case_str:
        return Shape(variables, vals, **optdata)
    elif 'unitcell' in lower_case_str:
        return UnitCellShape(variables, vals, **optdata)
    elif 'fourier' in lower_case_str:
        return FourierShape(variables, vals, **optdata)
    elif 'zernike' in lower_case_str:
        return ZernikeShape(variables, vals, **optdata)
    else:
        raise ValueError('invalid shape type %s' % shape_type)
        
def shape_factory_from_coords(neighbor_coords, ref_shape):
    """ Factory function to create a shape given coordinates.
        Use a reference shape as a template.
        
    @param neighbor_coords List of Coord objects for neighboring points
        
    @param ref_shape Shape object of appropriate type for template
        
    @retval Shape object of appropriate type
    
    """
    shape = ref_shape.copy()
    shape.build_from_coords(neighbor_coords)
    return shape