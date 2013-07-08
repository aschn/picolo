"""
@package shapedb
@author Anna Schneider
@version 0.1
@brief Contains class for ShapeDB
"""

# import from standard library
import xml.dom.minidom as dom
import xml.etree.ElementTree as ET
import collections

# import modules in this package
from shapes import shape_factory_from_values
from classifiers import classifier_factory

class ShapeDB:    
    """ Class that provides an interface to the Shape prototypes
            and Classifier.
        Current implementation stores the data on disc as a xml file
            and in memory as a dict.
        Supported Shape types are ['Fourier', 'Zernike', 'UnitCell', 'Generic'].
        Supported Classifier algorithms are ['SVM', 'GMM', 'None'].
        
        Generic xml structure:
            
            <data shape-type="string" classifier-type="string">
            
                <class-shape name="string">
                
                    <component type="string">number</component>
                    
                    <floatvar type="string">number</floatvar>

                    <boolvar type="string">bool</floatvar>
                    
                    <textvar type="string">string</textvar>
                    
                    <p>comment string</p>
                    
                </class-shape>
                
            </data>
        
    """
    def __init__(self, fname = ''):
        """ Constructor.
        
        @param self The object pointer
        
        @param fname String for filename that points to xml file that
            defines shape prototypes and/or classifier.
            
        """
        # initialize shape data info to default/null values
        self._data = dict()
        self._shape_names = []
        self.null_shape_name = ''

        # initialize classifier info to default/null values
        self._classifier = classifier_factory()
        
        # read from file if given
        if '.xml' in fname:
            print 'reading from', fname
            self.load(fname)
            
                    
    def __getitem__(self, n):
        return self._data[n]
        
    def __len__(self):
        return len(self._data)
        
    def __iter__(self):
        for sn in self.names():
            yield self[sn]
        
    def names(self):
        """ Returns a list of strings of valid shape names.
        """
        return self._shape_names
        
    def class_names(self):
        """ Returns a list of strings of valid class names, i.e.,
            all shape names plus the name of the null class (default is '').
        """
        return [self.null_shape_name] + self._shape_names
        
    def add(self, name, fshape):
        """ Add a new named shape.
        
        @param name String for shape name
        
        @param fshape Shape object
        
        """
        self._data[name] = fshape  
        self._shape_names.append(name)          
                    
    def discard(self, name):
        """ Discard a named shape.
        """
        if name in self._data:
            del self._data[name]
            del self._shape_names[self._shape_names.index(name)]
            
    def shape_type(self):
        """ Returns a string with the name of the stored Shape types.
            Defaults to 'Generic' if there are multipe types present.
            
        """
        types = set([shape.get('type') for shape in self._data.values()])
        if len(types) == 1:
            return types.pop()
        else:
            return 'Generic'
        
    def compute_match(self, name, features):
        """ Get goodness of fit for features to shape name using classifer.
        
        @param self The object pointer
        
        @param name String for shape name
        
        @param features Shape object to be matched
        
        @retval Float between 0 and 1 for degree or probability of match
        
        """
        # forward request to classifier
        return self._classifier.compute_match(self._data[name], features)

    def is_match(self, name, val):
        """ Get existence of good fit for features to shape name using classifer.
        
        @param self The object pointer
        
        @param name String for shape name
        
        @param val Number for goodness of fit (ie result of compute_match)
        
        @retval Bool for validity of match
        
        """
        # forward request to classifier
        return self._classifier.is_match(self._data[name], val)

    def save(self, xmlf):
        """ Saves the current state of the database to an xml file on disc.
            Overwrites any existing content in the file.
        
        @param xmlf String with path to file
        
        """
        # start a new tree
        root = ET.Element('data')
            
        # set shape and classifier types
        root.set('shape-type', self.shape_type())
        root.set('classifier-type', self._classifier.algorithm())

        # process shapes to add
        for name in self.names():
            root.append(self._shape2xml(name))

        # write back to file, via minidom
        f = open(xmlf, 'wb')
        f.write(self._prettyprint(root))
        f.close()
        
        # return success
        return True
        
    def load(self, xmlf):
        """ Read the contents of an xml file into the database.
        
        @param xmlf String with path to xml file

        """
        # load xml doc into tree
        tree = ET.parse(xmlf)
        root = tree.getroot()
        
        # set up classifier with cutoff
        classifier_type = root.get('classifier-type')
        cutoff = root.get('cutoff', default=0)
        self._classifier = classifier_factory(classifier_type, cutoff)
            
        # set up shape type
        shape_type = root.get('shape-type')

        # loop over shapes in file
        for shape_elt in root.findall('./class-shape'):
                
            # get mandatory data
            name = shape_elt.get('name')
            variables = [self._parse_component_name(shape_type, comp.get('type'))
                            for comp in shape_elt.findall('./component')]
            vals = [float(comp.text) for comp in shape_elt.findall('./component')]
            
            # get optional data
            optdata = dict()
            optdata['type'] = shape_type
            for child in shape_elt.findall('./*'):
                key = child.get('type').strip()
                data = self._parse_elt(child)
                optdata[key] = data

            # create and add shape
            shape = shape_factory_from_values(shape_type, variables,
                                              vals, optdata)
            self.add(name, shape)
            
    def _parse_elt(self, elt):
        """ Parse xml elements based on the tag.
            If the tag is listvar, boolvar, floatvar, or csvvar,
            parse as that type.
            Anything else is treated as text.
        
        @param elt xml element from ElementTree
        
        @retval data Data in elt.text or its children
        
        """
        if elt.tag in ['listvar']: # process recursively
            data = [self._parse_elt(child) for child in elt.findall('./*')]
            
        elif elt.tag in ['boolvar']: # process as boolean
            data = bool(elt.text)
            
        elif elt.tag in ['floatvar']: # process as float
            data = float(elt.text)
            
        elif elt.tag in ['csvvar']: # process as comma-separated list
            tmpdata = elt.text.split(',')
            data = []
            for val in tmpdata:
                if "'" in val:
                    data.append(val.strip("'"))
                else:
                    data.append(float(val))
                    
        else: # process as text
            data = elt.text.strip()
        
        # return
        return data

    def _shape2xml(self, name):
        """ Set up xml DOM element for shape name.
        
        @param name String for shape name
        
        @retval xml element
        
        """
        # set up xml element
        elt = ET.Element('class-shape')
        elt.set('name', name)
        
        # add components
        for var, val in self._data[name].iter_components():
            comp = ET.Element('component')
            comp.set('type', str(var))
            comp.text = str(val)
            elt.append(comp)

        # add other attributes
        for var, val in self._data[name].iter_params():
            
            # check type of variable
            if isinstance(val, bool):
                child = ET.Element('boolvar')
            elif isinstance(val, int) or isinstance(val, float):
                child = ET.Element('floatvar')
            elif isinstance(val, str):
                child = ET.Element('textvar')
            elif isinstance(val, collections.Iterable):
                child = ET.Element('csvvar')
                val = ','.join([str(v) for v in val])
            else:
                msg = 'Parameter %s of shape %s is %s. ' % (str(var), name,
                                                            type(val))
                msg += 'Should a bool, number, string, or iterable.'
                raise TypeError(msg)
                
            # finish making element
            child.set('type', str(var))
            child.text = str(val)
            elt.append(child)
                
        # return
        return elt

    def _prettyprint(self, elt):
        """ Return printable version of element.
        """
        uglystring = ET.tostring(elt)
        prettydom = dom.parseString(uglystring)
        return prettydom.toprettyxml()
        
    def _parse_component_name(self, shape_type, name_string):
        """ Parse component name string from xml based on shape type.
        
        @param self The object pointer
        
        @param shape_type String for shape type
        
        @param name_string String from <component name="string">number</component>
        
        @retval variable type:
                String for UnitCell or Generic,
                int for Fourier,
                tuple (int, int) for Zernike
                
        """
        if shape_type is 'UnitCell' or 'Generic':
            return name_string
            
        elif shape_type is 'Fourier':
            return int(name_string)

        elif shape_type is 'Zernike':
            return tuple(int(i) for i in name_string.strip('()').split(',')) 
            
        else:
            raise ValueError("invalid shape type %s." % shape_type)
        