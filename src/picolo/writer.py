"""
@package picolo
@author Anna Schneider
@version 0.1
@brief Contains class for Writer
"""

# import from standard library
import os

# import external packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

class Writer:
    """ Class that handles plotting and output for Matcher.
        Uses matplotlib for plotting.
    """
    ##
    # @var matcher
    # @brief Matcher object to output info from
    
    def __init__(self, matcher):
        """ Constructor.

        @param self The object pointer
        
        @param matcher Matcher object
        
        """
        self.matcher = matcher            
    
    def _add_mask(self, ax):
        # draw mask if present
        if self.matcher.mask:
            ax.imshow(self.matcher.mask.mask_for_show, aspect='equal',
                       cmap=cm.gray, extent=self.matcher.mask.extent)
    
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
        fig.suptitle('best matches for %s' % (os.path.basename(self.matcher.name)))
        ax = fig.gca()
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        
        # set up labels and colors
        nshapes = len(self.matcher.shapes.class_names())
        possible_colors = ['white', 'red', 'blue', 'green', 'purple',
                           'orange', 'yellow', 'brown', 'pink', 'gray']
        class_colors = possible_colors[0:nshapes]

        # compute and store values
        class_labels = self.matcher.classify()
        print zip(self.matcher.shapes.class_names(), np.bincount(class_labels))
               
        # plot locations and order parameters
        for ishape in range(nshapes):
            xs = np.asarray(self.matcher.config.x)[class_labels==ishape]
            ys = np.asarray(self.matcher.config.y)[class_labels==ishape]
            ax.scatter(xs, ys, s=50, c=class_colors[ishape],
                        label=self.matcher.shapes.class_names()[ishape])
        ax.legend(bbox_to_anchor=(1.35, 1))
        ax.axis(self.matcher.mask.extent)

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
                     (os.path.basename(self.matcher.name), shapename))
        
        # get subplot layout
        if self.matcher.shapes.shape_type() == 'UnitCell':
            n_features = 3
        else:
            raise ValueError('invalid shape mode %s' % self.matcher.shapes.shape_type())

        # set up data
        pltnames = ['a (nm)', 'b (nm)', 'angle (deg)']
        target_vals = [self.matcher.shapes[shapename].get('a'),
                       self.matcher.shapes[shapename].get('b'),
                       self.matcher.shapes[shapename].get('degrees')]
        data = [ [], [], [] ]
        uc_xs = []
        uc_ys = []
        other_xs = []
        other_ys = []
        for ip in self.matcher.config.indices:
            uc = self.matcher.get_features(shapename, ip)
            if uc.get('is_valid'):
                data[0].append(uc.get('a'))
                data[1].append(uc.get('b'))
                data[2].append(uc.get('degrees'))
                uc_xs.append(self.matcher.config.x[ip])
                uc_ys.append(self.matcher.config.y[ip])  
            else:
                other_xs.append(self.matcher.config.x[ip])
                other_ys.append(self.matcher.config.y[ip]) 
        minmaxes = []
        for i in range(n_features):
            try:
                absmax = np.max(np.abs(np.asarray(data[i]) - target_vals[i]))
            except ValueError: # if no ucs are valid
                absmax = 0
            minmaxes.append(absmax)
        
        # plot matches 
        for iplot in range(n_features):
            # plot data
            ax = fig.add_subplot(2, 3, iplot+1)
            ax.set_title(pltnames[iplot])
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')

            normalizer = colors.Normalize(vmin=target_vals[iplot]-
                                            minmaxes[iplot],
                                          vmax=target_vals[iplot]+
                                            minmaxes[iplot])
            ax.scatter(other_xs, other_ys, c='gray', s=10)
            sc = ax.scatter(uc_xs, uc_ys, c=data[iplot], cmap=cm.RdBu_r, s=10,
                        norm=normalizer)
            try:
                fig.colorbar(sc, ax=ax)
            except TypeError: # if no ucs are valid
                pass
            ax.axis([0, self.matcher.config.Lx, 0, self.matcher.config.Ly])

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
        ax = fig.gca()
        ax.set_xlabel('x position')
        ax.set_ylabel('y position')
        
        # plot triangles
        for ip,ns in neighbor_list.iteritems():
            for jp in ns:
                edge = [ip, jp]
                x_i = self.matcher.config.ximages[edge]
                y_i = self.matcher.config.yimages[edge]
                ax.plot(x_i, y_i, 'k')
        ax.axis(self.matcher.mask.extent)

        # draw mask
        self._add_mask(ax)
        
        # save plot to file
        self._save_fig(fig, fname)
            