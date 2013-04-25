"""
@package mask
@author Anna Schneider
@version 0.1
@brief Contains classes for Mask
"""

# import from standard library
import sys
import collections
import os

# import external packages
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

class Mask:
    """Class that uses an image to mask coordinates
        and define boundary conditions.

    Invalid regions are defined as pixels with 0 or negative values.
    Pixel resolution must be equal in x and y dim
    Assumes no periodic boundary conditions.

    """
    ##    
    # @var mask 
    # @brief ndarray of bools, True inside valid regions, False outside
    #
    # @var cf_px2nm 
    # @brief Float for conversion factor of pixels to nm
    #
    # @var cf_nm2px 
    # @brief Float for conversion factor of nm to pixels
    #                                 
    # @var boundary_kdtree 
    # @brief scipy.spatial.KDTree of points on boundary
    #
    # @var px_to_edge 
    #
    # @var extent 
    # @brief Tuple of numbers, (xmin, xmax, ymin, ymax)
                                 
    def __init__(self, imfile, Lx, Ly):
        """public: initialize mask
        
        @param self The object pointer
        
        @param imfile String for filename of image (type=tiff, etc)
        
        @param Lx Number for box side length in nm in x dim
        
        @param Ly Number for box side length in nm in y dim
        
        """
        # read image into array
        # x = 2nd dim, y = 1st dim
        # origin = upper left for imshow
        im = plt.imread(imfile)

        # convert to Cartesian coordinate form
        # x = 1st dim, y = 2nd dim
        # origin = lower left for indexing
        newim = im.copy()
        self.im = np.swapaxes(newim, 0,1)
        self.mask = self.im > 0

        # if nothing is masked out, return empty
        if np.count_nonzero(self.mask) == 0:
            return None

        # set up images with origin for nice imshow behavior
        self.im_for_show = im[::-1]
        self.mask_for_show = self.im_for_show > 0

        # set distance to pixel conversions
        self._set_units(Lx, Ly)

        # test
        self._test(Lx, Ly)

        # look for file with precomputed edge info
        self.edgefile = os.path.splitext(imfile)[0] + "_pxToEdge.txt"

        if os.path.isfile(self.edgefile):
            # read pxToEdge from file
            self.px_to_edge = np.genfromtxt(self.edgefile, unpack=True)
            print "read pixel edge info from file", self.edgefile

        else:
            # don't set up edge masks yet
            self.px_to_edge = None

    def _test(self, Lx, Ly):
        """ Test class cf_methods.
        
        @param self The object pointer
        
        @param Lx Number for box side length in nm in x dim
        
        @param Ly Number for box side length in nm in y dim
        
        """
        try:
            assert((0,self.mask.shape[1]-1) == self.point_nm2px(0, Ly-1e-5))
        except AssertionError:
            print "ERROR in self.point_nm2px(0, Ly)"
            print self.point_nm2px(0, Ly), Ly, self.mask.shape[1]-1
            sys.exit()

        try:
            assert((0,0) == self.point_nm2px(0, 0))
        except AssertionError:
            print "ERROR in self.point_nm2px(0, 0)"
            print self.point_nm2px(0, 0)
            sys.exit()

    def _set_units(self, Lx, Ly):
        """ Set box size and unit conversions.

        @param self The object pointer
        
        @param Lx Number for box side length in nm in x dim
        
        @param Ly Number for box side length in nm in y dim
        
        """
        # test Lx and Ly
        assert(Lx > 0)
        assert(Ly > 0)

        # find x and y conversions
        x_px2nm = float(self.mask.shape[0]) / float(Lx)
        x_nm2px = float(Lx) / float(self.mask.shape[0])
        y_px2nm = float(self.mask.shape[1]) / float(Ly)
        y_nm2px = float(Ly) / float(self.mask.shape[1])

        # test for equality within rounding error
        assert(abs(x_px2nm - y_px2nm) < 1e-4)
        assert(abs(x_nm2px - y_nm2px) < 1e-4)

        # set conversion factors
        self.cf_px2nm = x_px2nm
        self.cf_nm2px = y_px2nm
        
        # set extent for image
        self.extent = (0, Lx, 0, Ly)

    def _point_nm2px(self, xc, yc):
        """ Convert coordinates in nm to coordinates in px.

        @param self The object pointer
        
        @param xc Number or iterable of x coords in nm in .box frame
        
        @param yc Number or iterable of y coords in nm in .box frame
        
        @retval xp Number or iterable of x coords in px in image frame
        
        @retval yp Number or iterable of y coords in px in image frame
        
        """
        # set up iterable bools
        x_isit = isinstance(xc, collections.Iterable)
        y_isit = isinstance(yc, collections.Iterable)

        # check that xc and yc are both iterable or both not
        assert(x_isit == y_isit)

        # if iterable...
        if x_isit:
            # use copy to retain iterable type
            xp = xc.copy()        
            yp = yc.copy()
            # convert each item
            for ip in range(len(xp)):
                xp[ip] = int(xc[ip] * self.cf_nm2px)
                yp[ip] = int(yc[ip] * self.cf_nm2px)
        else:
            # convert single items
            xp = int(xc * self.cf_nm2px)
            yp = int(yc * self.cf_nm2px)

        # return
        return xp, yp
            
    def _point_px2nm(self, xp, yp):
        """ Convert coordinates in px to coordinates in nm.

        @param self The object pointer
        
        @param xp Number or iterable of x coords in px in image frame
        
        @param yp Number or iterable of y coords in px in image frame
        
        @retval xc Number or iterable of x coords in nm in .box frame
        
        @retval yc Number or iterable of y coords in nm in .box frame
        
        """
        # set up iterable bools
        x_isit = isinstance(xp, collections.Iterable)
        y_isit = isinstance(yp, collections.Iterable)

        # check that xc and yc are both iterable or both not
        assert(x_isit == y_isit)

        # if iterable...
        if x_isit:
            # use copy to retain iterable type
            xc = xp.copy()        
            yc = yp.copy()
            # convert each item
            for ic in range(len(xc)):
                xc[ic] = float(xp[ic] * self.cf_px2nm)
                yc[ic] = float(yp[ic] * self.cf_px2nm)
        else:
            # convert single items
            xc = xp * self.cf_px2nm
            yc = yp * self.cf_px2nm
        
        # return
        return xc, yc
        
    def _set_edges(self):
        """ Compute pixel masks that eliminate edge regions.

        @param self The object pointer
        
        """
        # set up distances to edges
        px_to_edge = np.zeros_like(self.mask, dtype=np.float)

        # set up boundary tree
        self._set_boundary()
        print "computing distances from %d valid pixels to %d edge pixels, this may take a bit..." % (
            np.count_nonzero(self.mask), self.boundary_kdtree.data.shape[0])

        # loop over True indices in mask
        for xp, yp in np.transpose(np.nonzero(self.mask)):
            # save distance from pixel to edge
            px_to_edge[xp,yp] = self.dist_to_edge(xp, yp, units='px')

        # store 
        self.px_to_edge = px_to_edge

        # test
        print "max dist from edge is %d px = %d nm" % (np.amax(self.px_to_edge), np.amax(self.pxToEdge)*self.cf_px2nm)

    def _set_boundary(self):
        """ Set up kd tree of mask boundary points.

        @param self The object pointer
        
        """
        # set up storage and initialize with image edges
        boundary_pairs = []
        for xp in range(0, self.mask.shape[0]):
            boundary_pairs.append([xp, -1])
            boundary_pairs.append([xp, self.mask.shape[1]])
        for yp in range(0, self.mask.shape[1]):
            boundary_pairs.append([-1, yp])
            boundary_pairs.append([self.mask.shape[0], yp])

        # loop over all invalid points
        for xp, yp in np.transpose(np.nonzero(self.mask == False)):
            # set up neighbors to right, left, top, and bottom
            inds = [[xp+1,yp], [xp-1,yp], [xp,yp+1], [xp,yp-1]]
            
            # loop over neighbors
            for nxp, nyp in inds:
                # if neighbor indices in array, test for edge
                if nxp >= 0 and nyp >= 0 and nxp < self.mask.shape[0] and nyp < self.mask.shape[1]:
                    if self.mask[nxp,nyp]:
                        boundary_pairs.append([xp,yp])
                        break
               
        # set up kd tree
        self.boundary_kdtree = spatial.KDTree(np.asarray(boundary_pairs))

    def write_edges(self, outfile=None):
        """ Write to file the distance from each pixel to the closest edge.
        
        @param self The object pointer
        
        @param outfile String with path to output file; default is self.edgefile
        
        """
        # set edge info 
        self._set_edges()        

        # choose outfile
        if not outfile:
            outfile = self.edgefile
        
        # write pxToEdge to file for later
        np.savetxt(outfile, self.px_to_edge, fmt="%4.3f")

    def dist_to_edge(self, xc, yc, units='nm'):
        """ Find distance from point to nearest boundary point.

        @param self The object pointer
        
        @param xc Number for x coord, in nm (unless units flag is different)
        
        @param yc Number for y coord, in nm (unless units flag is different)
        
        @param units String that specifies units for input and output,
            can be 'nm' or 'px'
            
        @retval Float for shortest distance to boundary,
            in nm (unless units flag is different)
            
        """
        # check units
        assert(units in ['nm', 'px'])

        # convert to pixels, all distance in pixels until end
        if units is 'nm':
            xp, yp = self._point_nm2px(xc, yc)
        else:
            xp, yp = xc, yc
            
        # if already boundary, return 0
        if not self.mask[xp,yp]:
            dist_px = 0

        # is pxToEdge set, use it
        elif self.px_to_edge is not None:
            dist_px = self.px_to_edge[xp,yp]

        # or just calculate from scratch
        else:
            # use KD tree to find distance from point to edge
            dist_px, id_at_edge = self.boundary_kdtree.query([xp,yp])

        # return in correct units
        if units == 'nm':
            return float(dist_px) * self.cf_px2nm
        else:
            return float(dist_px)

    def is_interior(self, xc, yc, cutoff_dist, units='nm'):
        """ Figure out if point is closer than cutoff_dist to edge.

        @param self The object pointer
        @param xc Number for x coord, in nm (unless units flag is different)
        
        @param yc Number for y coord, in nm (unless units flag is different)
        
        @param cutoff_dist Float for width of edge region to discard,
            in nm (unless units flag is different)
            
        @param units String that specifies units for input and output,
            can be 'nm' or 'px'
        @retval Bool that's True if interior, False if edge
        
        """
        # check units
        assert(units in ['nm', 'px'])

        # test distance
        if self.dist_to_edge(xc, yc, units) > cutoff_dist:
            return True
        else:
            return False

    def is_valid_seg(self, x1, y1, x2, y2):
        """ Test if line segment between (x1,y1) and (x2,y2)
            is inside the boundaries.

        @param self The object pointer
        
        @param x1 Number for x coord of first point, in nm
        
        @param y1 Number for y coord of first point, in nm
        
        @param x2 Number for x coord of second point, in nm
        
        @param y2 Number for y coord of second point, in nm
        
        @retval Bool, True if line segment is inside,
            False if crosses a boundary
            
        """
        # convert to pixels, all distance in pixels until end
        xp1, yp1 = self._point_nm2px(x1, y1)
        xp2, yp2 = self._point_nm2px(x2, y2)

        # if endpoints are already boundary, return False
        if not np.all(self.mask[[xp1,xp2],[yp1,yp2]]):
            return False

        # find relative positions of points
        xsign = np.sign(xp2 - xp1)
        ysign = np.sign(yp2 - yp1)

        # set up line segment, y=f(x)
        if xp2 != xp1:
            slope = float(yp2 - yp1) / float(xp2 - xp1)
            xs = np.arange(xp1, xp2, xsign)
            fys = slope * (xs - xp1) + yp1
            # round ys to integers
            ys = np.array(np.rint(fys), dtype=int)
        # or x=f(y)
        else:
            slope = float(xp2 - xp1) / float(yp2 - yp1)
            ys = np.arange(yp1, yp2, ysign)
            fxs = slope * (ys - yp1) + xp1
            # round xs to integers
            xs = np.array(np.rint(fxs), dtype=int)

        # test if line segment is outside mask
        if not np.all(self.mask[xs,ys]):
            # if outside, return False
            return False
        else: 
            # if inside return True
            return True

    def get_area(self, cutoff_dist=None):
        """ Compute and return unmasked (valid) area.

        @param self The object pointer
        
        @param cutoff_dist Float for width of edge region to discard, in nm
        
        @retval valid area, in nm^2
        
        """
        # set up counter
        n = 0

        if cutoff_dist:
            # count valid pixels in the correct edge mask
            n = np.count_nonzero(self.px_to_edge > cutoff_dist)
        else:
            # count valid pixels in full mask
            n = np.count_nonzero(self.mask)

        # convert to number of pixels to area
        area = float(n) * self.cf_px2nm**2

        # return
        return area
        
