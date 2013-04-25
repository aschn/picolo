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

class Coord:
    """ Cartesian and polar coordinates for given (x,y) pair.
        theta is in range (-pi, pi), degrees is in range(-180,180)
    """    
    ##
    # @var x
    # @brief Float for x coord (Cartesian)
    #
    # @var y 
    # @brief Float for y coord (Cartesian)
    #
    # @var r 
    # @brief Float for r coord (polar)
    #
    # @var theta 
    # @brief Float for theta coord in radians (polar)
    #
    # @var degrees
    # @brief Float for theta coord in degrees (polar)
    
    def __init__(self, dx, dy):
        """ Constructor. Calculates (r, theta) given (x,y).
        
        @param dx Float for x coord
        
        @param dy Float for y coord
        
        """
        self.x = float(dx)
        self.y = float(dy)
        self.r = math.sqrt(self.x*self.x + self.y*self.y)
        if self.r > 0:
            c = dx / self.r
            s = dy / self.r
            self.theta = math.atan2(s,c)
        else:
            self.theta = 0.0
        self.degrees = math.degrees(self.theta)
            
    def __repr__(self):
        """ Returns a printable string. """
        retstr = "(x,y) = (%2.1f, %2.1f); " % (self.x,self.y)
        degs = np.remainder(self.degrees, 360.0)
        retstr += "(r,theta) = (%2.1f, %3f)" % (self.r,degs)
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
