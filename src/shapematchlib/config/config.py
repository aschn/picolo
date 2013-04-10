"""
@module config
@author Anna Schneider
Contains classes for Config, NeighborList, Coord
"""

# import from standard library
import sys
import csv
import math

# import external packages
import numpy as np

####################################################

class Config:
    """ Class for handling and analyzing 2D particle coordinates.
    
    Attributes:
        
    @var x Ndarray of floats for x coords (in nm)
    
    @var y Ndarray of floats for y coords (in nm)
    
    @var N Int for number of points
    
    @var indices range(self.N)
    
    @var doPBC Bool for  whether to assume periodic boundary conditions
    
    @var Lx Float for box width
    
    @var Ly Float for box height
    
    @var ximages Ndarray of all images of particle x coordinates
        
    @var yimages Ndarray of all images of particle y coordinates
    
    """
    
    def __init__(self, xc, yc, pbc = False, lx = 0, ly = 0):
        """ Initialize box around coordinates,
            with or without periodic boundaries.

        @param self The object pointer
        
        @param xc Iterable of x coords (in nm)
        
        @param yc Iterable of y coords (in nm)
        
        @param pbc Bool for whether to assume periodic boundary conditions,
            default False
            
        @param lx Float for box width, default 0 (must set if pbc=True)
        
        @param ly Float for box height, default 0 (must set if pbc=True)
        
        @param ximages         
        
        """
        # set coords        
        assert(len(xc) == len(yc))
        self.x = np.asarray(xc)
        self.y = np.asarray(yc)      
        self.N = len(xc)
        self.indices = range(self.N)
       
        # get max and min coords
        xmin = np.min(self.x)
        xmax = np.max(self.x)
        ymin = np.min(self.y)
        ymax = np.max(self.y)
    
        # set up box parameters
        self.doPBC = pbc
        if lx == 0:
            self.Lx = xmax
        else:
            self.Lx = lx
        if ly == 0:
            self.Ly = ymax
        else:
            self.Ly = ly

        try:
            assert(xmin >= 0 and ymin >= 0 and xmax <= self.Lx and ymax <= self.Ly)
        except AssertionError:
            print "***"
            print "ERROR: box size is bad"
            print "min not nonnegative? xmin = %f, ymin = %f" % (xmin, ymin)
            print "or max too big? xmax = %f for Lx = %f, ymax = %f for Ly = %f" % (xmax, self.Lx, ymax, self.Ly)
            sys.exit()

        # set up periodically replicated images if using PBC
        if self.doPBC:
            self.ximages, self.yimages = self._get_periodic_images()
        else:
            self.ximages, self.yimages = self.x, self.y

        # log
        print "initialized box with PBC = %s, Lx = %d, Ly = %d" % (self.doPBC,
                                                                   self.Lx,
                                                                   self.Ly)
            
##########################
# PRIVATE UTILITY METHODS
##########################                

    def _PBC_dist(self, x1, y1, x2, y2):
        """ Compute distance between two points,
            taking periodic boundary conditions into account.

        @param self The object pointer
        
        @param x1 Number for x coord of point 1, in nm
        
        @param y1 Number for y coord of point 1, in nm
        
        @param x2 Number for x coord of point 2, in nm
        
        @param y2 Number for y coord of point 2, in nm
        
        @retval dist Float distance between points
        
        """
        # get differences between coords without PBC
        dx = x2 - x1
        dy = y2 - y1
    
        # if PBC, loop over x and y
        if self.doPBC:
            for (dc, c1, c2, Lc) in [ (dx, x1, x2, self.Lx) , (dy, y1, y2, self.Ly)]:
                
                # reset difference between coords
                halfL = Lc * 0.5
                if dc > halfL:
                    dc -= Lc
                elif dc > -halfL:
                    continue
                else:
                    dc += Lc

        # return Euclidian distance
        return math.sqrt(dx*dx + dy*dy)
        
    def _get_periodic_images(self):
        """ Set up periodic images of particle coordinates.

        order of periodic images: \n
        1 2 3 \n
        4 0 5 \n
        6 7 8 \n

        @param self The object pointer
        
        @retval ximages Ndarray of all images of particle x coordinates
        
        @retval yimages Ndarray of all images of particle y coordinates
        
        """
        ximages = np.resize(self.x, self.N*9)
        yimages = np.resize(self.y, self.N*9)
        image = 0
        for yimage in [1,0,-1]:
            for ximage in [-1,0,1]:
                if ximage == 0 and yimage == 0:
                    continue
                else:
                    image += 1
                    for ip in self.indices:
                        image_ip = ip + image * self.N
                        ximages[image_ip] = self.x[ip] + ximage * self.Lx
                        yimages[image_ip] = self.y[ip] + yimage * self.Ly
        return ximages, yimages

    def _is_tri_in_image(self, p0, p1, p2):
        """ Determine if three vertices form a triangle that is in the
            'minimum image'.
        
        order of periodic images: \n
        1 2 3 \n
        4 0 5 \n
        6 7 8 \n
        
        Resulting indices will be in images 0,1,2,4 only,
            with at least one point in 0 if possible.
        
        @param self The object pointer
        
        @param p0 Int for index of first vertex
        
        @param p1 Int for index of second vertex
        
        @param p2 Int for index of third vertex
        
        @retval Bool, True if triangle fits the criteria, False if not
        
        """
        # set up tri
        tri = sorted([p0, p1, p2])
        
        # set up current images and indices within images
        images = [ip / int(self.N) for ip in tri]

        # if all vertices are already in images 0,1,2,4
        if set(images).issubset(set([0,1,2,4])):
            # and if at least 1 vertex in image 0, or if images = [1,2,4]
            if 0 in images or set(images) == set([1,2,4]):
                # then it's good
                return True
        else:
            return False

    def _interior_particles(self, mask, cutoff_dist = None):
        """ Compute number of particles at least cutoff_dist away from edge.

        @param self The object pointer
        
        @param mask Mask object
        
        @param cutoff_dist Float for max radial distance to consider, in nm
        
        @retval Int for number of interior particles
        
        """
        if cutoff_dist > 0:
            n = 0
            for ip in range(self.N):
                if mask._is_interior(self.x[ip], self.y[ip], cutoff_dist):
                    n += 1
            return n
        else:
            return self.N

##########################
# PUBLIC UTILITY METHODS
##########################                

    def coord_at(self, ip):
        """ Get Coord object for a particle.
        
        @param ip Int for particle id
        
        @retval Coord object
        
        """
        return Coord(self.x[ip], self.y[ip])

    def density(self, mask = None, cutoff_dist = None):
        """ Compute particle density per nm^2.

        @param self The object pointer
        
        @param mask Mask object
        
        @param cutoff_dist Float for max radial distance to consider, in nm
        
        @retval Float for particle density
        
        """
        # use mask if available and asked for
        if mask is not None:

            n = self._get_interior_particles(mask, cutoff_dist=cutoff_dist)
            area = mask.get_area(cutoff_dist)
            try:
                return float(n) / area
            except ZeroDivisionError:
                return 0

        # otherwise, use whole box area
        else:
            return float(self.N) / self.Lx / self.Ly


##########################
# ANALYSIS METHODS
##########################                

    def radial_distribution(self, fname, cutoff_dist = 60.0,
                            binwidth = 1.0, mask = None, ips = None):
        """ Compute g(r) and print to file (space-delimited csv).

        Periodic boundary conditions used based on self.doPBC. \n
        Mask used based on availability and usemask.

        output file format: \n
        col1 distance, in nm \n
        col2 g(r) at that distance \n
        col3 counts at that distance \n
 
        @param self The object pointer
        
        @param fname String for filename to output g(r) to
        
        @param cutoff_dist Number for max radial distance to consider, in nm
        
        @param binwidth Number for histogram bin size, in nm
        
        @param mask Mask object
        
        """
        # set up outfile
        outfile = csv.writer(open(fname, 'wb'), delimiter=' ')

        # set up storage
        maxr = min(self.Lx * 0.5, self.Ly * 0.5, cutoff_dist)
        bins = [i*binwidth for i in range(int(maxr/binwidth))]
        counts = np.zeros(len(bins))
        g_norm = np.zeros(len(bins))
        n_particles = 0.0

        # reset usemask based on mask availability and PBC
        usemask = (mask is not None) and (not self.doPBC)
        
        # find ips
        if not ips:
            ips = self.indices

        # loop over particles
        for ip in ips:

            # if mask, pass if too close to edge
            if usemask and ips == self.indices:
                if not mask.is_interior(self.x[ip], self.y[ip], maxr):
                    continue
        
            # if ok for reference particle, count it
            n_particles += 1

            # loop over all other particles
            for jp in range(self.N):
                # don't store distance to yourself
                if ip == jp:
                    continue
                
                # get distance between particles, with or without PBC
                dist = self._PBC_dist(self.x[ip], self.y[ip], self.x[jp], self.y[jp])

                # add to histogram
                if dist < maxr:
                    for ibin in reversed(range(len(bins))):
                        if dist > bins[ibin]:
                            counts[ibin] += 1
                            break
                    
        # normalize
        area_in_bin = [math.pi*(bin+binwidth)*(bin+binwidth) - 
                       math.pi*bin*bin for bin in bins]
        if usemask and ips == self.indices:
            bg_particles_per_area = self.density(mask, cutoff_dist=maxr)
        elif mask is not None:
            bg_particles_per_area = self.density(mask, cutoff_dist=0)
        else:
            bg_particles_per_area = self.density(None, cutoff_dist=0)
        g_norm =  counts / area_in_bin / bg_particles_per_area / n_particles
        
        # write to file
        print "writing to", fname
        for ibin in range(len(bins)):
            outfile.writerow([bins[ibin]+0.5*binwidth,
                              g_norm[ibin], counts[ibin]])

    def nearest_neighbor_distribution(self, fname, cutoff_dist = 35.0,
                                      binwidth = 1.0, mask = None,
                                      ips = None):
        """ Compute histogram of nearest-neighbor distances
            and print to file (space-delimited csv).

        Periodic boundary conditions used based on self.doPBC. \n
        Mask used based on availability and usemask.

        output file format: \n
        col1 distance, in nm \n
        col2 probability of NN distance \n
        col3 counts at that distance \n
 
        @param self The object pointer
        
        @param fname String for filename to output to
        
        @param cutoff_dist Number for min distance to edge to consider, in nm
        
        @param binwidth Number for histogram bin size, in nm
        
        @param mask Mask object
        
        """
        # set up outfile
        outfile = csv.writer(open(fname, 'wb'), delimiter=' ')

        # set up storage
        maxr = min(self.Lx * 0.5, self.Ly * 0.5, cutoff_dist)
        bins = [i*binwidth for i in range(int(maxr/binwidth))]
        counts = np.zeros(len(bins))
        prob_norm = np.zeros(len(bins))

        # reset usemask based on mask availability and PBC
        usemask = (mask is not None) and not self.doPBC

        # find ips
        if not ips:
            ips = self.indices

        # loop over particles
        for ip in ips:

            # if mask, pass if too close to edge
            if usemask:
                if not mask.is_interior(self.x[ip], self.y[ip], maxr):
                    continue
        
            # if ok for reference particle, reset min distance
            min_dist = 2.0 * cutoff_dist

            # loop over all other particles
            for jp in range(self.N):
                # don't store distance to yourself
                if ip == jp:
                    continue
                
                # get distance between particles, with or without PBC
                dist = self._PBC_dist(self.x[ip], self.y[ip],
                                      self.x[jp], self.y[jp])

                # test if closer than current min
                if dist < min_dist:
                    # reject if connecting line cuts through mask boundary
                    if usemask:
                        if not mask.is_valid_seg(self.x[ip], self.y[ip],
                                                 self.x[jp], self.y[jp]):
                            continue
                    # store as min if ok
                    min_dist = dist

            # add to histogram
            for ibin in reversed(range(len(bins))):
                if min_dist > bins[ibin]:
                    counts[ibin] += 1
                    break
                    
        # normalize
        prob_norm =  counts / max(np.sum(counts), 1.0)
        
        # write to file
        print "writing to", fname
        for ibin in range(len(bins)):
            outfile.writerow([ bins[ibin]+0.5*binwidth, prob_norm[ibin],
                               counts[ibin] ])


####################################################

class Coord:
    """ Cartesian and polar coordinates for given (x,y) pair.
        Theta is in range (-pi, pi).

    Attributes:
        
    @var x Float for x coord (Cartesian)
    
    @var y Float for y coord (Cartesian)
    
    @var r Float for r coord (polar)
    
    @var theta Float for theta coord (polar)
    
    """
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
            
    def __repr__(self):
        """ Returns a printable string 
        """
        retstr = "(x,y) = (%2.1f, %2.1f); " % (self.x,self.y)
        degs = math.degrees(np.remainder(self.theta, 2.0*math.pi))
        retstr += "(r,theta) = (%2.1f, %3f)" % (self.r,degs)
        return retstr
        
####################################################

