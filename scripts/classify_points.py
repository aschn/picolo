#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author Anna Schneider
@version 0.1
@brief Typical script using picolo to classify local order of point particles
"""

import picolo
import argparse
import os.path as path
import time

# start timer
start = time.time()

# parse command-line arguments
brief = 'Typical script using picolo to classify local order of point particles.'
parser = argparse.ArgumentParser(description=brief)
parser.add_argument('filename', type=str, help='path to xy coord file')
parser.add_argument('tifname', type=str, help='path to image mask file')
parser.add_argument('xmlname', type=str, help='path to xml shape file')
parser.add_argument('Lx', type=float, help='width of field of view (same units as xy coords)')
parser.add_argument('Ly', type=float, help='height of field of view (same units as xy coords)')
args = parser.parse_args()

# set up file paths
rootname, ext = path.splitext(args.filename)
dirname = path.dirname(args.filename)

# set up matcher and writer
matcher = picolo.Matcher(args.filename, delim=' ', lx=args.Lx, ly=args.Ly,
                         imname=args.tifname, xmlname=args.xmlname,
                         name=rootname)
writer = picolo.Writer(matcher)

# draw classification graphic
writer.draw_classification(rootname+'_match_to_best.pdf')

# append fraction of particles and area that match each class to log
writer.write_fraction_particles_matched(path.join(dirname,
                                                  'xtal_particles_log.dat'))
writer.write_fraction_area_matched(path.join(dirname,
                                             'xtal_area_log.dat'))

# write features to file
writer.write_features(matcher.shapes.class_names()[-1], rootname+'_features.dat')

# end timer
end = time.time()
print 'Done with %s ... took %d seconds.' % (rootname, end-start)
