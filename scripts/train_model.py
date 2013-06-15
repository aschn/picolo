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
         Assumes input files are like output from \
         picolo.Writer.write_features()."
parser = argparse.ArgumentParser(description=brief)
parser.add_argument('--files', nargs='*', type=str,
                    help='path(s) to feature file(s)')
parser.add_argument('--type', choices=['gmm'], type=str,
                    help='type of classifier algorithm')
parser.add_argument('--xcols', nargs='*', type=int,
                    help='columns (after 0th) to use as features')
args = parser.parse_args()

# set up trainer
trainer = picolo.trainer_factory(args.type)

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

# choose best number of components by bootstrap
ks = range(1, 8)
n_reps = 200
bics = np.zeros((n_reps, len(ks)))
means = np.zeros([n_reps, trainer.n_classes, trainer.n_features])
for ik, k in enumerate(ks):
    for irep, (model, inds) in enumerate(trainer.bootstrap(n_reps=n_reps,
                                                           n_classes=k)):
        bics[irep, ik] = model.bic(inds)

# plot bootstrap results
plt.figure()
plt.boxplot(bics, positions=ks)

bics = [trainer.fit(n_classes=k).bic() for k in ks]
best_k = ks[np.argmin(bics)]
plt.figure()
plt.plot(ks, bics)
print "best k = %d" % best_k

# fit model
trainer.fit(n_classes=best_k)

# print to screen
if args.type == 'gmm':
    print "means:", trainer.means()
    print "sds:", trainer.sds()
    
# like R pairs, colored by predicted class label
predicted_labels = trainer.predict()
plt.figure()
for iplot, xarr, yarr in trainer.pairs():
    plt.subplot(trainer.n_features, trainer.n_features, iplot)
    plt.scatter(xarr, yarr, c=predicted_labels, cmap=cm.Set1, vmax=8)
#plt.show()

# bar chart by source, colored by predicted class label
plt.figure()
bar_lefts = np.arange(trainer.n_classes)
for isource in range(trainer.n_sources):
    predicted_labels_for_source = trainer.predict(isource)
    predicted_counts = np.bincount(predicted_labels_for_source,
                                   minlength=(trainer.n_sources))
    plt.subplot(1, trainer.n_sources, isource+1)
    patches = plt.bar(bar_lefts, predicted_counts)
    for ibar in range(len(bar_lefts)):
        patches[ibar].set_facecolor(cm.Set1(ibar/8.0))
    plt.xlim([bar_lefts[0], bar_lefts[-1]+1])
plt.show()
        
# end timer
end = time.time()
print 'Done ... took %d seconds.' % (end-start)
