"""
@package picolo
@author Anna Schneider
@version 0.1
@brief Contains class for Matcher
"""

# import from standard library
import csv
import os
import logging

# import external packages
import numpy as np

# import modules in this package
from config import Config, DistNeighbors, DelaunayNeighbors, Mask
from shapes import ShapeDB, shape_factory_from_coords

class Matcher():
    """Class that computes and classifies 2D crystal types.
    """
    ##
    # @var name
    # @brief String for self-identification
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
    def __init__(self, fname, xcol=0, ycol=1, delim=' ',
                 tagcol=None, goodtag=None, trainingcol=None,
                 header=0, pbc=False, lx=0, ly=0,
                 imname=None, skipneg=True,
                 name='', xmlname=''):
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
                
        """
        # set up name
        self.name = name

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
        if '_features' not in dir(self):
            self._features = dict()
        if shapename:
            namelist = [shapename]
        else:
            namelist = self.shapes.names()
        
        # loop through shape names
        for sname in namelist:
            self._features[sname] = []
            
            # loop through particles
            for ip in self.config.indices:
                
                # compute features for particle
                particle_features = self.get_features(sname, ip)
                    
                # store features
                self._features[sname].append(particle_features)
                                
    def get_features(self, shapename, particle_id):
        """ Get features for a particle using params from a shape.
        
        @param self The object pointer
        
        @param shapename String for a shape in self.shapes
        
        @param particle_id Int for a particle id
        
        @retval Shape object containing features
        
        """
        # return what's stored in self.features if available
        try:
            shape = self._features[shapename][particle_id]
            
        # or compute if not
        except (KeyError, IndexError):
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
        
    def feature_matrix(self, shapename, normalize=False):
        """ Get feature matrix for all particles using params from a shape.
        
        @param self The object pointer
        
        @param shapename String for a shape in self.shapes
        
        @param normalize Bool for whether or not to normalize each row,
            default False
                
        @retval features Ndarray containing features, shape
            n_particles x n_components
        
        """
        # set up storage
        m_features = np.zeros([self.config.N, len(self.shapes[shapename])])
        
        # add data
        for ip in self.config.indices:
            shape = self.get_features(shapename, ip)
            if shape.get('is_valid'):
                v_features = shape.get_vals(normalize)
                m_features[ip] = v_features
                
        # return
        return m_features
    
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
        if shapename not in self._features.keys():
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
            (0 or 1 if filter is on, float if filter is off)
        
        """
        if shapename is 'all':
            class_names = self.shapes.class_names()
            if do_smoother and not do_filter:
                msg = 'Nonsensical to have get_best_match use smoother '
                msg += 'without filter if shapename==all. '
                msg += 'Turning on filter...'
                do_filter = True
                raise RuntimeWarning(msg)
                
        else:
            try:
                class_names = [self.shapes.null_shape_name for i in
                                self.shapes.class_names()]
                index = self.shapes.class_names().index(shapename)
                class_names[index] = shapename            
            except IndexError:
                raise KeyError('Invalid shape name %s.' % shapename)
            
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
        logging.debug(match_vals)
        
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
        logging.debug(match_vals)

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
            
    def classify(self, shapename='all', particle_id='all'):
        """ Classify particles. This is a wrapper around get_best_match
            and is recommended over calling get_best_match directly in most
            cases. See docs for get_best_match for more details.
        
        @param self The object pointer
        
        @param shapename String 'all' for one-vs-rest classification into
                any class (including the null class), 
            or a shape name for one-vs-one classification vs the null class.

        @param particle_id String 'all' to classify all particles,
            or int to classify 1 particle
            
        @retval class_labels Ndarray of ints in that are categorical labels
            for each particle (null class is 0)
        
        """
        # get matches
        best_names, is_matches = self.get_best_match(shapename=shapename,
                                                      particle_id=particle_id,
                                                      do_filter=True,
                                                      do_smoother=True)
                                                      
        # convert into range(len(self.shapes.shape_names())
        class_labels = np.asarray([self.shapes.class_names().index(best_names[i])
                                   for i in range(len(is_matches))],
                                   dtype=int)
        
        # return
        return class_labels
                
    def areas_matched(self, class_labels):
        """ Calculates the area of particles that match each non-null class,
            based on the provided class labels.
            
        @param self The object pointer
        
        @param class_labels Ndarray of ints for class labels (0 is null class)
                
        @retval areas Ndarray of floats with nm^2 of area matched,
            size is # of shapes
        
        """
        # set up storage
        areas = np.zeros(len(self.shapes.names()))
        
        # each UnitCell shape knows its area, so use it
        if self.shapes.shape_type() is 'UnitCell':
            for ishape, n_matched in enumerate(self.count_matched(class_labels)):
                uc_area = self.shapes[self.shapes.names()[ishape]].area()
                logging.debug('%s area %0.2f' % (self.shapes.names()[ishape],
                                                 uc_area))
                areas[ishape] = float(n_matched) * uc_area

        # or use Delaunay triangulation to estimate enclosed area
        else:    
            for ishape, sn in enumerate(self.shapes.names()):
                ids = self.particles_matched(class_labels, sn)
                if len(ids) > 0:
                    area, edges = self.delaunay_neighbors.area_of_point_set(ids,
                                                                    self.config.x[ids],
                                                                    self.config.y[ids])
                    areas[ishape] = area
                     
        # return
        return areas
        
    def count_matched(self, class_labels):
        """ Gets the number of particles that match each non-null class,
            based on the provided class labels.
        
        @param self The object pointer
        
        @param class_labels Ndarray of ints for class labels (0 is null class)
                
        @retval counts Ndarray of ints for number of particles matched,
            size is # of shapes
        
        """
        classcounts = np.bincount(class_labels,
                                  minlength=len(self.shapes.class_names()))
        return classcounts[1:]

    def particles_matched(self, class_labels, shapename='all'):
        """ Gets the ids of particles that match each class, based on
            the provided class labels. If shapename is 'all', gets particles
            that match any class.
            
        @param self The object pointer
        
        @param class_labels Ndarray of ints for class labels (0 is null class)
                        
        @retval ndarray of ints for ids of particles matched
                
        """        
        # pull out matching ids
        if shapename == 'all':
            return np.where(class_labels)[0]
        else:
            index = self.shapes.class_names().index(shapename)
            return np.where(class_labels==index)[0]
        