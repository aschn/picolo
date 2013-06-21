#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Anna Schneider
@version 0.1
@brief Typical script using picolo to extract features from point particles
"""

import picolo
from shapes import shape_factory_from_values

import argparse
import os.path as path
import time
import csv

# start timer
start = time.time()

# parse command-line arguments
brief = 'Typical script using picolo to extract features from point particles.'
parser = argparse.ArgumentParser(description=brief)
parser.add_argument('filename', type=str, help='path to xy coord file')
parser.add_argument('shape', type=str, help='type of features to extract',
                    choices=['UnitCell', 'Fourier', 'Zernike'])
parser.add_argument('dist', type=float, help='distance cutoff to neighbors')
parser.add_argument('--train', action='store_true',
                    help='include flag to only get features for prespecified training rows')
args = parser.parse_args()

# set up file paths
rootname, ext = path.splitext(args.filename)
dirname = path.dirname(args.filename)

# set up matcher
matcher = picolo.Matcher(args.filename, delim=' ', name=rootname,
                         trainingcol=2)

# create and add default shape of correct type
shape = shape_factory_from_values(args.shape,
                                  optdata={'neighbor_dist': args.dist,
                                           'max_dist': args.dist})
matcher.shapes.add('test', shape)

# get ndarray of features and particle ids by comparing to 'test' shape
features = matcher.feature_matrix('test')

# open csv writer
outfile = '%s_%s_features.dat' % (rootname, args.shape)
writer = csv.writer(open(outfile, 'w'), delimiter=' ')
                    
# write header
writer.writerow(['id'] + shape.get_components())

# loop over particle ids
if args.train:
    inds = matcher.training_ids
else:
    inds = range(matcher.config.N) 
for ip in inds:
    
    # only write features for particles with valid shapes
    if matcher.get_features('test', ip).get('is_valid'):
        
        # write row of features
        writer.writerow([ip] + ['%0.4f' % x for x in features[ip]])

# end timer
end = time.time()
print 'Done with %s ... took %d seconds.' % (rootname, end-start)
