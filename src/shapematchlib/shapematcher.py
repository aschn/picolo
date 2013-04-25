"""
@package shapematchlib
@author Anna Schneider
@version 0.1
@brief Contains classes for ShapeMatcher and ShapeMatchWriter
"""

# import from standard library
import csv
import os

# import external packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# import modules in this package
from config import Config, DistNeighbors, DelaunayNeighbors, Mask
from shapes import ShapeDB, shape_factory_from_coords

class ShapeMatcher():
    """Class that computes and classifies 2D crystal types.
    """
    ##
    # @var name
    # @brief String for self-identification
    #
    # @var mode
    # @brief String for the shape type
    #
    # @var config
    # @brief Config object containing coordinates
    #
    # @var training_ids
    # @brief List of ints for training coords
    #
    # @var mask
    # @brief Mask object containing image mask
    #
    # @var shapes
    # @brief ShapeDB object containing target shapes to match against
    #
    # @var features
    # @brief Dict containing features for each data point
    #    (key = shapename, val = List of features)

    def __init__(self, fname, xcol=0, ycol=1, delim=' ',
                 tagcol=None, goodtag=None, trainingcol=None,
                 header=0, pbc=False, lx=0, ly=0,
                 imname=None, skipneg=True,
                 name='', xmlname='', mode='UnitCell'):
        """ Constructor.
        
        @param self The object pointer
        
        @param fname String of filename to read coordinates from (csv formatted)
        
        @param xcol Int for column in fname containing x coordinates (0 indexed)
        
        @param ycol Int for column in fname containing y coordinates (0 indexed)
        
        @param delim String of csv delimiter in fname
        
        @param tagcol Int for column in fname containing tag values (0 indexed)
        
        @param goodtag Value in tagcol corresponding to rows to keep
        
        @param trainingcol Int for column in fname containing training values (0 indexed)
        
        @param header Int for number of header lines to skip, default 0
        
        @param skipneg Bool for whether to discard points at negative
            x or y coordinates
            
        @param pbc Bool for whether to assume periodic boundary conditions,
            default False
            
        @param lx Float for box width, default 0 (must set if pbc=True)
        
        @param ly Float for box height, default 0 (must set if pbc=True)
        
        @param imname String for filename of image for masking, default None
        
        @param name String for self-identification
        
        @param xmlname String with path to xml file defining shapes
        
        @param mode String for the shape type (default UnitCell)
        
        """
        # set up name
        self.name = name

        # set up shape matching mode
        self.mode = mode
        
        # set up config
        xc, yc, training_ids = self.read_config(fname, xcol, ycol, delim,
                                                tagcol, goodtag, trainingcol,
                                                header, skipneg)
        self.config = Config(xc, yc, pbc, lx, ly)
        self.training_ids = training_ids
        print "%d coordinates read from %s" % (self.config.N, fname)        

        # set up mask
        if imname and not self.config.doPBC:
            self.initialize_mask(imname, lx, ly)
        else:
            self.remove_mask()

        # set up neighbors
        self.dist_neighbors = DistNeighbors(self.config, mask=self.mask)
        self.delaunay_neighbors = DelaunayNeighbors(self.config, self.mask)

        # set up target shapes
        self.initialize_shapes(xmlname)
        
        # set up particle features
        self.set_features()
        
    def read_config(self, fname, xcol, ycol, delim, tagcol, goodtag,
                   trainingcol, header, skipneg):
        """ Read coordinates from csv file fname.
        
        @param fname String of filename to read coordinates from (csv formatted)
        
        @param xcol Int for column in fname containing x coordinates (0 indexed)
        
        @param ycol Int for column in fname containing y coordinates (0 indexed)
        
        @param delim String of csv delimiter in fname
        
        @param tagcol Int for column in fname containing tag values (0 indexed)
        
        @param goodtag Value in tagcol corresponding to rows to keep
        
        @param trainingcol Int for column in fname containing training values
            (0 indexed)
        @param header Int for number of header lines to skip, default 0
        
        @param skipneg Bool for whether to discard points at negative
            x or y coordinates
            
        @retval xc List of floats for x coordinates
        
        @retval yc List of floats for y coordinates
        
        @retval training_ids List of ints for ids of training coords
        
        """
        # set up infile
        infile = csv.reader(open(fname, 'rb'), delimiter=delim)
        
        # read infile
        xc = []
        yc = []
        training_ids = []
        for irow, row in enumerate(infile):
            if irow < header:
                # skip if header
                continue
            elif tagcol is not None and row[tagcol] != goodtag:
                # skip if this row has the wrong tag
                continue
            elif '#' in ' '.join(row):
                # skip if commented out
                continue                
            elif (float(row[xcol]) < 0 or float(row[ycol]) < 0) and skipneg:
                # skip if negative
                continue
            else:
                # store otherwise
                xc.append(float(row[xcol]))
                yc.append(float(row[ycol]))

                # store training ids
                if trainingcol is not None and len(row) - 1 >= trainingcol:
                    if int(row[trainingcol]):
                        training_ids.append( len(xc) - 1 )
        
        # return               
        return xc, yc, training_ids
               
    def initialize_shapes(self, fname = ''):
        """ Initialize the target crystal shapes from a file.
        
        @param self The object pointer
        
        @param fname String with path to shape file
        
        """        
        if os.path.isfile(fname):
            self.shapes = ShapeDB(fname)
        else:
            self.shapes = ShapeDB()
        
        # log
        print "initialized %d shapes from %s" % (len(self.shapes), fname)

    def initialize_mask(self, imfile, Lx, Ly):
        """ Initialize mask from image file.
        
        @param self The object pointer     
        
        @param imfile String with path to image  file(type=tiff, etc)
        
        @param Lx Number for box side length in nm in x dim
        
        @param Ly Number for box side length in nm in y dim
        
        """
        # log
        print "initializing mask from file", imfile

        # set up mask object
        self.mask = Mask(imfile, Lx, Ly)

    def remove_mask(self):
        """ Remove mask object.
        
        @param self The object pointer
        
        """
        self.mask = None
        
    def set_features(self, shapename = None):
        """ Set features for a shape.
        
        @param self The object pointer        
        
        @param shapename String for a shape in self.shapes. If None,
            set features for all shapes.
            
        """
        # set up
        if 'features' not in dir(self):
            self.features = dict()
        if shapename:
            namelist = [shapename]
        else:
            namelist = self.shapes.names()
        
        # loop through shape names
        for sname in namelist:
            self.features[sname] = []
            
            # loop through particles
            for ip in self.config.indices:
                
                # compute features for particle
                particle_features = self.get_features(sname, ip)
                    
                # store features
                self.features[sname].append(particle_features)
                                
    def get_features(self, shapename, particle_id):
        """ Get features for a particle using params from a shape.
        
        @param self The object pointer
        
        @param shapename String for a shape in self.shapes
        
        @param particle_id Int for a particle id
        
        @retval Shape object containing features
        
        """
        # return what's stored in self.features if available
        try:
            return self.features[shapename][particle_id]
            
        # or compute if not
        except KeyError or IndexError:
            # get ref shape
            ref_shape = self.shapes[shapename]
            
            # collect coords
            neighbor_ids = self.dist_neighbors.neighbors_within(ref_shape.get('max_dist'),
                                                                particle_id) 
            own_coord = self.config.coord_at(particle_id)
            neighbor_coords = [self.config.coord_at(ip) - own_coord
                                for ip in neighbor_ids]
                                                         
            # compute features based on mode
            shape = shape_factory_from_coords(neighbor_coords, ref_shape)
                
            # return
            return shape
    
    def get_raw_match(self, shapename, particle_id = 'all'):
        """ Get match for particle(s) to a shape.
        
        @param self The object pointer
        
        @param shapename String for a shape in self.shapes
        
        @param particle_id Int for a particle id.
            If 'all', get match for all particles in config.
            
        @retval List of floats for match to each particle
        """
        # check particle_id arg
        if particle_id not in ['all'] + self.config.indices:
            raise ValueError('invalid particle id %s' % str(particle_id))
        else:
            if particle_id is 'all':
                ip_list = self.config.indices
            else:
                ip_list = [particle_id]
                
        # check shape name arg
        if shapename is self.shapes.null_shape_name:
            # match is 0 to null shape
            return [0 for ip in ip_list]
        elif shapename not in self.shapes.names():
            raise ValueError('invalid shape name %s' % shapename)
            
        # set features, if not done so
        if shapename not in self.features.keys():
            self.set_features(shapename)
            
        # compute match
        retvals = []
        for ip in ip_list:
            features_for_ip = self.get_features(shapename, ip)
            val = self.shapes.compute_match(shapename, features_for_ip)
            retvals.append(val)
        
        # return 
        return retvals

    def get_best_match(self, shapename = 'all',
                       particle_id = 'all',
                       do_filter = True,
                       do_smoother = True):
        """ Get best match for particle(s) to shapename(s) using a pipeline.
            
            (1) Get raw match.
            
            (2) Get the best of the raw matches, if shapename is 'all'.
            
            (3) Apply an optional rejection cutoff filter.
            
            (4) Apply an optional smoother via a spatial k-nearest-neighbor
                filter (median if input is float, majority if input is bool).
                
        @param self The object pointer
        
        @param shapename String for a shape name.
            If 'all', get match to all shapes.
            
        @param particle_id Int for a particle id.
            If 'all', get match for all particles in config.
            
        @param do_filter Bool to turn on or off rejection filter
        
        @param do_smoother Bool to turn on or off k-NN smoother
        
        @retval match_names List of strings for shape names
        
        @retval match_vals List of numbers for goodness of match to shape
        
        """
        if shapename is 'all':
            class_names = self.shapes.class_names()
            if do_smoother and not do_filter:
                msg = 'Nonsensical to have get_best_match use smoother '
                msg += 'without filter if shapename==all. '
                msg += 'Turning on filter...'
                do_filter = True
                raise RuntimeWarning(msg)
                
        elif shapename in self.shapes.names():        
            class_names = [self.shapes.null_shape_name for i in
                            self.shapes.class_names()]
            index = self.shapes.class_names().index(shapename)
            class_names[index] = shapename
            
        else:
            raise KeyError('invalid shape name %s' % shapename)
            
        if particle_id is 'all':
            vals = np.zeros([len(class_names), self.config.N])
        else:
            vals = np.zeros([len(class_names), 1])
            
        # step 1: get raw match vals for all shapes
        for isn, sname in enumerate(class_names):
            if sname:
                vals[isn,:] = np.asarray(self.get_raw_match(sname,
                                                            particle_id))
            
        # step 2: best = argmax
        # this has the correct behavior for shapename=='all' or otherwise
        match_names = [class_names[i] for i in np.argmax(vals, axis=0)]
        match_vals = np.max(vals, axis=0)
        
        # step 3: apply match cutoff
        if do_filter:
            best_match_names = []
            match_bools = []
            for i in range(len(match_vals)):
                try:
                    if self.shapes.is_match(match_names[i], match_vals[i]):
                        best_match_names.append(match_names[i])
                        match_bools.append(1)
                    else:
                        best_match_names.append('')
                        match_bools.append(0)
                except KeyError:
                    best_match_names.append('')
                    match_bools.append(0)
            match_names = best_match_names
            match_vals = match_bools

        # step 4: smooth
        if do_smoother:
            if do_filter:
                # majority filter on shape, categorically
                presmooth_inds = [class_names.index(name) for name in match_names]
                postsmooth_inds = self.dist_neighbors.kNN_filter(presmooth_inds,
                                                                 self.config.indices,
                                                                 mode='mode',
                                                                 ownweight=1)
                match_names = [class_names[i] for i in postsmooth_inds]
                match_vals = [0 if class_names[i] is '' else 1 for i in postsmooth_inds]
            else:
                # median filter on value, only sensical if shapename != 'all'
                match_vals = self.dist_neighbors.kNN_filter(match_vals,
                                                            self.config.indices,
                                                            mode='median',
                                                            ownweight=1)

        # returnmatch
        return match_names, match_vals
            
    def classify(self, particle_id = 'all'):
        """ Classify particles using pipeline in self.get_best_match.
        
        @param self The object pointer
        
        @param particle_id String 'all' to classify all particles,
            or int to classify 1 particle
            
        @retval list of ints in that are categorical labels for each particle
        
        """
        # get matches
        best_names, is_matches = self.get_best_match(shapename = 'all',
                                                      particle_id = particle_id,
                                                      do_filter = True,
                                                      do_smoother = True)
                                                      
        # convert into range(len(self.shapes.names_with_null())
        class_labels = [self.shapes.class_names().index(best_names[i])
                        for i in range(len(is_matches))]
        
        # return
        return class_labels
                
    def fraction_matched(self, shapename = None):
        """ Calculates the fraction of particles that match a particular
            shape, or any shape if shapename not provided.
            TO DO: implement for non-UnitCell shape types!
            
        @param self The object pointer
        
        @param shapename String for the shape to check
        
        @retval ndarray of floats with fraction(s) of particles matched
        
        @retval ndarray of floats with fraction(s) of area matched
        
        """
        # only implemented for UnitCell now
        if self.mode is not 'UnitCell':
            raise RuntimeError('fraction_matched is only supported for shape mode UnitCell')
        
        # compute for a particular shape
        if shapename:
            best_names, is_matches = self.get_best_match(shapename,
                                                         particle_id = 'all')
            n_matched = np.count_nonzero(np.asarray(is_matches))
            uc_area = self.shapes[shapename].area()
            area_matched = float(n_matched) * uc_area
            
        # compute for all shapes
        else:
            class_labels = self.classify()
            n_matched = np.bincount(class_labels)[1:]
            uc_areas = [self.shapes[sn].area() for sn in self.shapes.names()]
            area_matched = np.multiply(n_matched, uc_areas)
                        
        # finish and return
        total_area = self.mask.get_area()
        total_n = float(self.config.N)
        return (n_matched / total_n, area_matched / total_area)
        
    def connected_components(self, props, ntype='dist'):
        """ Find distribution of connected component cluster sizes
            where property = True.
            Connectivity uses specified neighbor list (default is distance).

        @param self The object pointer
        
        @param props List of objects for truth testing,
            length = self.config.N
            
        @param ntype String for neighbor list, either 'dist' (default)
            or 'delaunay'
            
        @retval clusters List (sorted) of sets of particle ids,
            one set per cluster
            
        @retval uncluster List (sorted) of particle ids that didn't match
            prop and are thus not in clusters
            
        """
        # test input
        try:
            assert(len(props) == self.config.N)
        except AssertionError:
            raise AssertionError('Length of props %d does not match the number of particles %d' %
                                (len(props), self.config.N))
        
        # set up storage
        clusters = [] # list of sets of particle ids
        is_clustered = [False for i in range(self.config.N)] # list of bools for each particle id
        stack = [] # list of ids of particles to process
        icluster = 0 # running index of working cluster id
        uncluster = [] # list of particle ids that don't match prop and are thus not in clusters

        # choose method
        if ntype=='delaunay':
            neighbors = self.delaunay_neighbors
        else:
            neighbors = self.dist_neighbors          

        # loop over reference particles
        for iref in range(self.config.N):

            # if doesn't match prop, don't try to cluster
            if not props[iref]:
                uncluster.append(iref)

            # if clustered, pass
            elif is_clustered[iref]:
                continue

            # if not clustered, try to cluster
            else:
                # add to stack
                stack.append(iref)       
                # start a new cluster
                clusters.append(set())

                # loop through stack and deplete
                while(len(stack)) > 0:

                    # add from stack to cluster
                    ip1 = stack.pop()
                    clusters[icluster].add(ip1)

                    # loop over neighbors
                    for ip2 in neighbors[ip1]:
                        # filter by property and clustering
                        if props[ip2] and not is_clustered[ip2]:
                            # add to stack
                            stack.append(ip2)
                            is_clustered[ip2] = True

                # finished this cluster, on to the next one
                icluster += 1

        # sort lists
        clusters.sort(key = lambda c:len(c))
        uncluster.sort()

        # return
        return clusters, uncluster        

####################################################
        
class ShapeMatchWriter:
    """ Class that handles plotting and output for ShapeMatcher.
        Uses matplotlib for plotting.
    """
    ##
    # @var sm
    # @brief ShapeMatcher object to output info from
    
    def __init__(self, sm):
        """ Constructor.

        @param self The object pointer
        
        @param sm ShapeMatcher object
        
        """
        self.sm = sm            
    
    def _add_mask(self, ax):
        # draw mask if present
        if self.sm.mask:
            ax.imshow(self.sm.mask.mask_for_show, aspect='equal',
                       cmap=cm.gray, extent=self.sm.mask.extent)
    
    def _save_fig(self, fig, fname):
        if fname:
            # save plot to file
            fig.savefig(fname, transparent = True)
            print "saved plot of best shapes to file %s" % (fname)
        else:
            # display plot on screen
            fig.show()              
    
    def draw_classification(self, fname = None):
        """ Plot classification shape labels for each particle in config,
            or save to file if fname is given.
        
        @param self The object pointer
        
        @param fname String for filename in which to save plot image
            (image not saved by default)
            
        """
        # set up plot
        fig = plt.figure()
        fig.suptitle('best matches for %s' % (os.path.basename(self.sm.name)))
        fig.add_xlabel('x position')
        fig.add_ylabel('y position')
        
        # set up labels and colors
        nshapes = len(self.sm.shapes.class_names())
        possible_colors = ['white', 'red', 'blue', 'green', 'purple',
                           'orange', 'yellow', 'brown', 'pink', 'gray']
        class_colors = possible_colors[0:nshapes]

        # compute and store values
        class_labels = self.sm.classify()
        print zip(self.sm.shapes.class_names(), np.bincount(class_labels))
               
        # plot locations and order parameters
        ax = fig.add_subplot(111)
        for ishape in range(nshapes):
            xs = np.asarray(self.sm.config.x)[class_labels==ishape]
            ys = np.asarray(self.sm.config.y)[class_labels==ishape]
            ax.scatter(xs, ys, s=50, c=class_colors[ishape],
                        label=self.sm.shapes.class_names()[ishape])
        ax.legend(bbox_to_anchor=(1.35, 1))
        ax.axis([0, self.sm.config.Lx, 0, self.sm.config.Ly])

        # draw mask
        self._add_mask(ax)
        
        # save plot to file
        self._save_fig(fig, fname)
        
    def draw_unitcell_diagnostics(self, shapename, fname = None):
        """ Create diagnostic plots for unit cell shapes.
        
        @param self The object pointer
        
        @param shapename String for shape name
        
        @param fname String for filename in which to save plot image
            (image not saved by default)
            
        """
        # set up plot
        fig = plt.figure()
        fig.suptitle('Features for match of %s to shape %s' % 
                     (os.path.basename(self.sm.name), shapename))
        fig.add_xlabel('x position')
        fig.add_ylabel('y position')
        
        # get subplot layout
        if self.sm.mode == 'UnitCell':
            n_features = 3
        else:
            raise ValueError('invalid shape mode %s' % self.sm.mode)

        # set up data
        pltnames = ['a (nm)', 'b (nm)', 'angle (deg)']
        target_vals = [self.sm.shapes[shapename].get('a'),
                       self.sm.shapes[shapename].get('b'),
                       self.sm.shapes[shapename].get('degrees')]
        data = [ [], [], [] ]
        uc_xs = []
        uc_ys = []
        other_xs = []
        other_ys = []
        for ip, uc in enumerate(self.sm.features[shapename]):
            if uc.get('is_cell'):
                data[0].append(uc.get('a'))
                data[1].append(uc.get('b'))
                data[2].append(uc.get('degrees'))
                uc_xs.append(self.sm.config.x[ip])
                uc_ys.append(self.sm.config.y[ip])  
            else:
                other_xs.append(self.sm.config.x[ip])
                other_ys.append(self.sm.config.y[ip])               
        minmaxes = [np.max(np.abs(np.asarray(data[i]) - target_vals[i]))
                    for i in range(n_features)]
        
        # plot matches 
        for iplot in range(n_features):
            # plot data
            ax = fig.add_subplot(2, 3, iplot+1)
            ax.set_title(pltnames[iplot])
            normalizer = colors.Normalize(vmin=target_vals[iplot]-
                                            minmaxes[iplot],
                                          vmax=target_vals[iplot]+
                                            minmaxes[iplot])
            ax.scatter(other_xs, other_ys, c='gray', s=10)
            ax.scatter(uc_xs, uc_ys, c=data[iplot], cmap=cm.RdBu_r, s=10,
                        norm=normalizer)
            fig.colorbar(ax=ax)
            ax.axis([0, self.sm.config.Lx, 0, self.sm.config.Ly])

            # draw mask if present
            self._add_mask(ax)
                
                
        # plot features
        ax = fig.add_subplot(2, 3, 4)
        ax.set_title('a vs b')
        ax.scatter(data[1], data[0])
        ax = fig.add_subplot(2,3,5)
        ax.set_title('a vs angle')
        ax.scatter(data[2], data[0])
        ax = fig.add_subplot(2,3,6)
        ax.set_title('b vs angle')
        ax.scatter(data[2], data[1])
        
        # save plot to file
        self._save_fig(fig, fname)
        
    def draw_neighbors(self, neighbor_list, fname = None):
        """ Draw neighbor list to screen,
            or save to file if fname is given.
        
        @param self The object pointer
        
        @param neighbor_list NeighborList object (or derived therefrom)
        
        @param fname String for filename in which to save plot image
            (image not saved by default)
        """
        # set up plot
        # set up plot
        fig = plt.figure()
        fig.suptitle('Neighbors')
        fig.add_xlabel('x position')
        fig.add_ylabel('y position')
        
        # plot triangles
        ax = fig.add_subplot(111)
        for ip,ns in neighbor_list.iteritems():
            for jp in ns:
                edge = [ip, jp]
                x_i = self.sm.config.ximages[edge]
                y_i = self.sm.config.yimages[edge]
                ax.plot(x_i, y_i, 'k')
        ax.axis([0, self.sm.config.Lx, 0, self.sm.config.Ly])

        # draw mask
        self._add_mask(ax)
        
        # save plot to file
        self._save_fig(fig, fname)
            