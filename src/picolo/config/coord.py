"""
@package coord
@author Anna Schneider
@version 0.1
@brief Contains class for Coord
"""

# import from standard library
import math

# import external packages
import numpy as np

class Coord(object):
    """ Cartesian and polar coordinates for given (x,y) pair.
        theta is in range (-pi, pi), degrees is in range(-180,180)
    """    
    ##
    # @var x
    # @brief Float for x coord (Cartesian)
    #
    # @var y 
    # @brief Float for y coord (Cartesian)
    
    def __init__(self, dx, dy):
        """ Constructor. Calculates (r, theta) given (x,y).
        
        @param dx Float for x coord
        
        @param dy Float for y coord
        
        """
        self.x = float(dx)
        self.y = float(dy)
        self._theta = None
        self._degrees = None
        self._r = None
        
    @property
    def r(self):
        """ Distance from the origin """
        if self._r is None:
            self._r = math.sqrt(self.x*self.x + self.y*self.y)
        return self._r
        
    @property
    def theta(self):
        """ Angle in radians for (r,theta) """
        if self._theta is None:
            if self.r > 0:
                c = self.x / self.r
                s = self.y / self.r
                self._theta = math.atan2(s,c)
            else:
                self._theta = 0.0
        
        # return        
        return self._theta
        
    @property
    def degrees(self):
        """ Angle in degrees for (r,theta) """
        if self._degrees is None:
            self._degrees = math.degrees(self.theta)
        return self._degrees
            
    def __repr__(self):
        """ Returns a printable string. """
        retstr = "(x,y) = (%2.1f, %2.1f); " % (self.x,self.y)
        degs = np.remainder(self.degrees, 360.0)
        retstr += "(r,theta) = (%2.1f, %3.1f)" % (self.r,degs)
        return retstr
        
    def __add__(self, other):
        """ Add two coords. 
        
        @param self The left-hand coord
        
        @param other The right-hand coord
        
        @retval Coord object containing sum
        
        """
        return Coord(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """ Subtract right-hand coord from left-hand. 
        
        @param self The left-hand coord
        
        @param other The right-hand coord
        
        @retval Coord object containing sum
        
        """
        return Coord(self.x - other.x, self.y - other.y)
