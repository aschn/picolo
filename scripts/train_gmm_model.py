#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Anna Schneider
@version 0.1
@brief Typical script using picolo to train a multiclass model
"""

import picolo
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
parser.add_argument('--files', nargs='*', type=str,
                    help='path(s) to feature file(s)')
parser.add_argument('--xcols', nargs='*', type=int,
                    help='columns (after 0th) to use as features')
args = parser.parse_args()

# set up trainer
trainer = picolo.trainer_factory('gmm')

# read and load data
for filename in args.files:
    
    # read frome file into array
    if args.xcols:
        X_data = np.genfromtxt(filename, skip_header=1,
                               usecols=args.xcols)
    else:
        filedata = np.genfromtxt(filename, skip_header=1)
        X_data = filedata[:,1:]
        
    # load
    trainer.load(X_data)

# hard-code var names
# TO DO read from file
varnames = ['a (nm)', 'b (nm)', 'angle (degrees)']

# fit model
trainer.fit(n_classes=args.k)
predicted_labels = trainer.predict()
raw_means = trainer.means()
raw_sds = trainer.sds()

# run bootstrap and collect data
n_reps = 300
means_bs = np.zeros([n_reps, args.k, trainer.n_features])
sds_bs = np.zeros([n_reps, args.k, trainer.n_features])
print "bootstrapping with %d reps..." % n_reps
bootstrapper = trainer.bootstrap_fit(n_reps=n_reps, n_classes=args.k,
                                     labels_true=predicted_labels)
for irep, (model, inds) in enumerate(bootstrapper):
    means_bs[irep] = model.means()
    sds_bs[irep] = model.sds()
    
# plot bootstrapped means
plt.figure()
colors = ['r', 'b', 'g', 'purple', 'orange', 'yellow']
for ifeat in range(trainer.n_features):
    plt.suptitle('Estimated cluster means (%d bootstrap reps)' % n_reps)
    plt.subplot(1, trainer.n_features, ifeat+1)
    plt.xlabel(varnames[ifeat])
    plt.ylabel('count')
    for ik in range(args.k):
        plt.hist(means_bs[:, ik, ifeat], histtype='stepfilled',
                 color=colors[ik], alpha=0.5)
        plt.scatter(raw_means[ik, ifeat], 0.2*n_reps, c=colors[ik])
    plt.ylim([0, n_reps/2.0])
plt.show()
   
# plot bootstrapped SDs
plt.figure()
for ifeat in range(trainer.n_features):
    plt.suptitle('Estimated cluster SDs (%d bootstrap reps)' % n_reps)
    plt.subplot(1, trainer.n_features, ifeat+1)
    plt.xlabel(varnames[ifeat])
    plt.ylabel('count (%d bootstrap reps)' % n_reps)
    for ik in range(args.k):
        plt.hist(sds_bs[:, ik, ifeat], histtype='stepfilled',
                 color=colors[ik], alpha=0.5)
        plt.scatter(raw_sds[ik, ifeat], 0.2*n_reps, c=colors[ik])
    plt.ylim([0, n_reps/2.0])
plt.show()
       
# like R pairs, colored by predicted class label
plt.figure()
for iplot, ix, xarr, iy, yarr in trainer.pairs():
    plt.subplot(trainer.n_features, trainer.n_features, iplot)
    plt.xlabel(varnames[ix])
    plt.ylabel(varnames[iy])
    plt.scatter(xarr, yarr, c=predicted_labels, cmap=cm.Set1, vmax=8)
plt.show()

# bar chart by source, colored by predicted class label
plt.figure()
bar_lefts = np.arange(trainer.n_classes)
for isource in range(trainer.n_sources):
    # compute counts for source
    predicted_labels_for_source = trainer.predict(isource)
    predicted_counts = np.bincount(predicted_labels_for_source,
                                   minlength=(trainer.n_sources))
                               
    # draw
    plt.subplot(1, trainer.n_sources, isource+1)
    patches = plt.bar(bar_lefts, predicted_counts, align='center')
    for ibar in range(len(bar_lefts)):
        patches[ibar].set_facecolor(cm.Set1(ibar/8.0))
    plt.xlim([bar_lefts[0]-0.5, bar_lefts[-1]+0.5]) 
plt.show()
        
# end timer
end = time.time()
print 'Done ... took %d seconds.' % (end-start)
