#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Anna Schneider
@version 0.1
@brief Typical script using picolo to train a multiclass model
"""

import picolo
from shapes import shape_factory_from_values, ShapeDB
import argparse
import os.path as path
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import sys

# start timer
start = time.time()

# parse command-line arguments
brief = "Typical script using picolo to train an unsupervised \
         multiclass model. \
         Assumes input files are like output from extract_features.py."
parser = argparse.ArgumentParser(description=brief)
parser.add_argument('-name', type=str,
                    help='job name (used for naming output files)')
parser.add_argument('-k', type=int,
                    help='number of classes to fit to')
parser.add_argument('-nboot', type=int,
                    help='number of bootstrap repetitions (recommend 1000)')
parser.add_argument('--files', nargs='*', type=str,
                    help='path(s) to feature file(s)')
parser.add_argument('--xcols', nargs='*', type=int,
                    help='columns (after 0th) to use as features')
args = parser.parse_args()

# set up trainer
trainer = picolo.trainer_factory('gmm')

# read and load data
source_names = []
for filename in args.files:
    
    try:
        # read from file into array
        if args.xcols:
            X_data = np.genfromtxt(filename, skip_header=1,
                                   usecols=args.xcols)
        else:
            filedata = np.genfromtxt(filename, skip_header=1)
            X_data = filedata[:,1:]
    except IOError:
        print "problem reading %s, skipping" % filename
        continue
        
    # load
    print "reading %d rows from file %s" % (X_data.shape[0], filename)
    trainer.load(X_data)
    source_names.append(path.splitext(path.basename(filename))[0])

# hard-code var names
# TO DO read from file
varnames = ['a (nm)', 'b (nm)', 'angle (degrees)']
template_shape = shape_factory_from_values('UnitCell',
                                  optdata={'neighbor_dist': 30.0,
                                           'max_dist': 30.0})
# configure matplotlib
mpl.rc('axes', labelsize='small', titlesize='medium')
mpl.rc('xtick', labelsize='small')
mpl.rc('ytick', labelsize='small')
mpl.rc('legend', fontsize='x-small')

###########################
# DATA MODEL  
###########################
  
# fit model
trainer.fit(n_classes=args.k)
data_labels = trainer.predict()
raw_means = trainer.means()
raw_sds = trainer.sds()

# save data model to xml
data_shapedb = ShapeDB()
for ishape, shape in enumerate(trainer.params_as_shapes(template_shape)):
    data_shapedb.add(str(ishape), shape)
data_shapedb.save('%s_k%d_data_db.xml' % (args.name, args.k))

# bar chart by source, colored by predicted class label
plt.figure()
plt.suptitle('classifications using data model for %s' % args.name)
bar_lefts = np.arange(trainer.n_classes)
for isource in range(trainer.n_sources):
    # compute counts for source
    predicted_labels_for_source = trainer.predict(isource)
    predicted_counts = np.bincount(predicted_labels_for_source,
                                   minlength=(trainer.n_classes))
                               
    # draw
    plt.subplot(1, trainer.n_sources, isource+1)
    plt.xlabel('cluster label')
    plt.ylabel('number of points')
    plt.title(source_names[isource])
    patches = plt.bar(bar_lefts, predicted_counts, align='center')
    for ibar in range(len(bar_lefts)):
        patches[ibar].set_facecolor(cm.Set1(ibar/8.0))
    plt.xlim([bar_lefts[0]-0.5, bar_lefts[-1]+0.5]) 
plt.savefig('%s_k%d_training_classes.pdf' % (args.name, args.k))
  
# like R pairs, colored by predicted class label
plt.figure()
plt.suptitle('pairwise features using data model for %s' % args.name)
for iplot, ix, xarr, iy, yarr in trainer.pairs():
    plt.subplot(trainer.n_features, trainer.n_features, iplot)
    plt.xlabel(varnames[ix])
    plt.ylabel(varnames[iy])
    plt.scatter(xarr, yarr,
                c=data_labels, cmap=cm.Set1, vmax=8, alpha=0.5)
plt.savefig('%s_k%d_pairs.pdf' % (args.name, args.k))  
  
###########################
# BOOTSTRAP
###########################  
  
# run bootstrap and collect data
n_reps = args.nboot
means_bs = np.zeros([n_reps, args.k, trainer.n_features])
sds_bs = np.zeros([n_reps, args.k, trainer.n_features])
print "bootstrapping with %d reps and %d classes..." % (n_reps, args.k)
bootstrapper = trainer.bootstrap_fit(n_reps=n_reps, n_classes=args.k,
                                     labels_true=data_labels)
for irep, (model, inds) in enumerate(bootstrapper):
    means_bs[irep] = model.means()
    sds_bs[irep] = model.sds()
    
# set trainer with expectation values of params
average_means = np.mean(means_bs, axis=0)
average_sds = np.mean(sds_bs, axis=0)
sd_means = np.std(means_bs, axis=0)
sd_sds = np.std(sds_bs, axis=0)
trainer.set_params(means=average_means, sds=average_sds)
bootstrap_labels = trainer.predict()

# save bootstrapped model to xml
bs_shapedb = ShapeDB()
for ishape, shape in enumerate(trainer.params_as_shapes(template_shape)):
    bs_shapedb.add(str(ishape), shape)
bs_shapedb.save('%s_k%d_bootstrap_db.xml' % (args.name, args.k))

# plot bootstrapped means and SDs
plt.figure()
plt.suptitle('Estimated cluster parameters (%d bootstrap reps)' % n_reps)
colors = ['r', 'b', 'g', 'purple', 'orange', 'yellow', 'brown', 'pink']
for ifeat in range(trainer.n_features):
    # means
    plt.subplot(2, trainer.n_features, ifeat+1)
    plt.xlabel('mean ' + varnames[ifeat])
    plt.ylabel('count')
    x_range = (means_bs[:,:,ifeat].min(), means_bs[:,:,ifeat].max())
    nbins = int(x_range[1] - x_range[0] + 1)
    while nbins < 16:
        nbins *= 2
    for ik in range(args.k):
        label = "%2.2f +/- %2.2f" % (average_means[ik,ifeat],
                                     sd_means[ik,ifeat])
        plt.scatter(raw_means[ik, ifeat], 0.2*n_reps, c=colors[ik])
        plt.hist(means_bs[:, ik, ifeat], bins=nbins, range=x_range,
                 histtype='stepfilled', color=colors[ik], alpha=0.5,
                 label=label)
    plt.gca().set_ylim(bottom=0)
    plt.legend()

    # sds
    plt.subplot(2, trainer.n_features, trainer.n_features+ifeat+1)
    plt.xlabel('SD ' + varnames[ifeat])
    plt.ylabel('count')
    x_range = (sds_bs[:,:,ifeat].min(), sds_bs[:,:,ifeat].max())
    nbins = int(x_range[1] - x_range[0] + 1)
    while nbins < 16:
        nbins *= 2
    for ik in range(args.k):
        label = "%2.2f +/- %2.2f" % (average_sds[ik,ifeat],
                                     sd_sds[ik,ifeat])
        plt.scatter(raw_sds[ik, ifeat], 0.2*n_reps, c=colors[ik])
        plt.hist(sds_bs[:, ik, ifeat], bins=nbins, range=x_range,
                 histtype='stepfilled', color=colors[ik], alpha=0.5,
                 label=label)
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    plt.legend()
plt.savefig('%s_k%d_parameters.pdf' % (args.name, args.k))
  
# confusion matrix
plt.matshow(trainer.confusion_matrix(data_labels), cmap=cm.Reds)
plt.colorbar()
plt.title('confusion matrix for %s' % args.name)
plt.xlabel('cluster label using bootstrapped model')
plt.ylabel('cluster label using data model')
plt.savefig('%s_k%d_confusion.pdf' % (args.name, args.k))
      
# end timer
end = time.time()
print 'Done ... took %d seconds.' % (end-start)
