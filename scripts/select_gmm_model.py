#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Anna Schneider
@version 0.1
@brief Typical script using picolo to sekect a multiclass model
"""

import picolo
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

# start timer
start = time.time()

# parse command-line arguments
brief = "Typical script using picolo to select an unsupervised \
         multiclass model. \
         Assumes input files are like output from \
         picolo.Writer.write_features()."
parser = argparse.ArgumentParser(description=brief)
parser.add_argument('-name', type=str,
                    help='job name (used for naming output files)')
parser.add_argument('-maxk', type=int,
                    help='max number of classes to select from')
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
    
# set up for bootstrapping
ks = range(1, args.maxk+1)
n_reps = args.nboot
bics_raw = np.zeros(len(ks))
bics_bs = np.zeros([n_reps, len(ks)])
count_selected = np.zeros(len(ks), dtype=int)
print "bootstrapping with %d reps..." % n_reps

# select models using bootstrap
select_bootstrapper = trainer.bootstrap_select(n_reps=n_reps, ks=ks)
for irep, (selected_k, bics_rep) in enumerate(select_bootstrapper):
    selected_ik = ks.index(selected_k)
    count_selected[selected_ik] += 1
    bics_bs[irep] = bics_rep

# analyze bootstrap results
median_bics_bs = np.median(bics_bs, axis=0)
best_k_median = ks[np.argmin(median_bics_bs)]
frac_bics_bs = count_selected / float(n_reps)
best_k_frac = ks[np.argmax(frac_bics_bs, axis=0)]

# get BICs without bootstrap
for ik in range(len(ks)):
    trainer.fit(n_classes=ks[ik])
    bics_raw[ik] = trainer.bic()
best_k_raw = ks[np.argmin(bics_raw)]

# plot bootstrap results
plt.figure()
plt.subplot(211)
plt.title('Model selection for %s (%d bootstrap reps)' % (args.name, n_reps))
plt.xlabel('$k$')
plt.ylabel('frequency $k$ is selected')
barcolors = ['white' if k != best_k_frac else 'black' for k in ks]
plt.bar(ks, frac_bics_bs, align='center', color=barcolors)
plt.xlim([ks[0]-0.5, ks[-1]+0.5])
plt.subplot(212)
plt.xlabel('$k$')
plt.ylabel('BIC')
plt.boxplot(bics_bs, positions=ks)
plt.scatter(ks, bics_raw, c='g', marker='o',
            label='value w/o bootstrap (best $k=%d$)' % best_k_raw)
plt.scatter(ks, median_bics_bs, c='r', marker='_',
            label='median from bootstrap (best $k=%d$)' % best_k_median)
plt.xlim([ks[0]-0.5, ks[-1]+0.5])
plt.legend(prop={'size':8})
plt.savefig('%s_selection.pdf' % args.name)

# end timer
end = time.time()
print 'Done ... took %d seconds.' % (end-start)
print best_k_frac, best_k_median, best_k_raw