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

# choose best number of components
ks = range(1,15)
bics = [trainer.fit(n_classes=k).bic() for k in ks]
best_k = ks[np.argmin(bics)]
print "best k = %d" % best_k
print "from BICs %s" % bics    

# fit model
trainer.fit(n_classes=best_k)

# output
if args.type == 'gmm':
    print trainer.means()
    print trainer.sds()
        
# end timer
end = time.time()
print 'Done ... took %d seconds.' % (end-start)
