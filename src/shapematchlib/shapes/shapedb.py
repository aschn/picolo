"""
@package shapes
@module shapes
@author Anna Schneider
Contains class for ShapeDB
"""

# import from standard library
import math
import xml.dom.minidom as dom
import xml.etree.ElementTree as ET

# import modules in this package
from shapes import UnitCellShape, ZernikeShape, FourierShape
from classifiers import SVMClassifier, GMMClassifier

class ShapeDB:    
    """ Class that provides an interface to the shape prototypes
            and classifier.
        Current implementation stores the data on disc as a xml file
            and in memory as a dict.
        Supported Shape types are ['Fourier', 'Zernike', 'UnitCell'].
        Supported Classifier types are ['SVM', 'GMM'].
        
    """
    def __init__(self, fname = ''):
        """ Constructor.
        
        @param self The object pointer
        
        @param fname String for filename that points to xml file that
            defines shape prototypes and/or classifier.
            
        """
        # initialize shape data info to default/null values
        self._data = dict()
        self._shape_constructors = {'Fourier': FourierShape,
                                   'Zernike': ZernikeShape,
                                   'UnitCell': UnitCellShape}
        self._shape_constructor = None
        self._shape_names = []
        self.null_shape_name = ''

        # initialize classifier info to default/null values
        self._classifier = None
        self._classifier_constructors = {'SVM': SVMClassifier,
                                        'GMM': GMMClassifier}
        
        # read from file if given
        if '.xml' in fname:
            print 'reading from', fname
            self.read_XML(fname)
            
                    
    def __getitem__(self, n):
        return self._data[n]
        
    def __len__(self):
        return len(self._data)
        
    def names(self):
        """ Returns a list of strings of valid shape names.
        """
        return self._shape_names
        
    def class_names(self):
        """ Returns a list of strings of valid class names, i.e.,
            all shape names plus the name of the null class (default is '').
        """
        return [self.null_shape_name] + self._shape_names

    def read_XML(self, xmlf):
        """ Read the contents of an xml file into the database.
        
        Generic structure:
            <data shape-type="string" classifier-type="string">
                <class-shape name="string">
                    <component type="string">number</component>
                    <floatvar type="string">number</floatvar>
                    <textvar type="string">string</textvar>
                    <p>comment string</p>
                </class-shape>
            </data>
        
        @param xmlf String with path to xml file

        """
        # load xml doc into tree
        tree = ET.parse(xmlf)
        root = tree.getroot()
        
        # set up classifier with cutoff
        classifier_type = root.get('classifier-type')
        cutoff = root.get('cutoff', default=0)
        self._classifier = self._classifier_constructors[classifier_type](cutoff)            
            
        # set up shape type
        shape_type = root.get('shape-type')
        self._shape_constructor = self._shape_constructors[shape_type]

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
                if child.tag in ['p', 'textvar']:
                    optdata[child.get('type').strip()] = child.text.strip()
                elif child.tag in ['floatvar']:
                    vtype = child.get('type').strip()
                    if 'theta' in vtype:
                        optdata[vtype] = math.radians(float(child.text))
                    else:
                        optdata[vtype] = float(child.text)

                    
            # create and add shape
            shape = self._shape_constructor(variables, vals, **optdata)
            self.add(name, shape)

    def _parse_component_name(self, shape_type, name_string):
        """ Parse component name string from xml based on shape type.
        
        @param self The object pointer
        
        @param shape_type String for shape type
        
        @param name_string String from <component name="string">number</component>
        
        @retval variable type:
                String for UnitCell,
                int for Fourier,
                tuple (int, int) for Zernike
                
        """
        if shape_type is 'UnitCell':
            return name_string
            
        elif shape_type is 'Fourier':
            return int(name_string)

        elif shape_type is 'Zernike':
            return tuple(int(i) for i in name_string.strip('()').split(',')) 
            
        else:
            msg = "Got invalid shape type %s." % shape_type
            msg += "Valid types are" + ", ".join(self._shape_constructors.keys())
            raise ValueError(msg)
        
    def update_XML(self, xmlf):
        """ Update an xml file on disc with the current state of the database.
        
        @param xmlf String with path to xml file
        
        """
        # load xml doc into tree
        doctree = ET.parse(xmlf)
        root = doctree.getroot()

        # categorize shapes to add, remove, or refresh (ie remove then add)
        xmlnames = set([shape.get('name')
                        for shape in root.findall('./class-shape')])
        datanames = set(self._data.keys())
        to_add = datanames - xmlnames
        to_remove = xmlnames - datanames
        to_refresh = datanames & xmlnames

        # process shapes to remove
        for shape in root.findall('./class-shape'):
            if shape.get('name') in to_remove | to_refresh:
                root.remove(shape)

        # process shapes to add
        for name in to_add | to_refresh:
            root.append(self.shape2xml(name))

        # write back to file, via minidom
        f = open(xmlf, 'wb')
        f.write(self.prettyprint(root))
        f.close()

    def add(self, name, fshape):
        """ Add a new named shape.
        
        @param name String for shape name
        
        @param fshape Shape object
        
        """
        self._data[name] = fshape  
        self.name_list.append(name)          
                    
    def discard(self, name):
        """ Implement dict.discard(key) interface.
        """
        if name in self._data:
            del self._data[name]
            del self._shape_names[self._shape_names.index(name)]

    def shape2xml(self, name):
        """ Set up xml DOM element for shape name.
        
        @param name String for shape name
        
        @retval xml element
        
        """
        # get type
        stype = self._data[name].type
        
        # set up xml element
        elt = ET.Element(stype)
        elt.set('name', name)
        
        # add text attributes
        for key, val in vars(self._data[name]).iteritems():
            if isinstance(val, str):
                if key not in ['type']:
                    child = ET.Element('p')
                    child.set('type', str(key))
                    child.text = val
                    elt.append(child)
                
        # add components
        for il in range(self.data[name].nls):
            comp = ET.Element('component')
            comp.set('l', str(self._data[name].ls[il]))
            comp.text = str(self._data[name].vals[il])
            elt.append(comp)

        # add numerical attributes
        for key, val in vars(self._data[name]).iteritems():
            if isinstance(val, int) or isinstance(val, float):
                if key not in ['nls', 'mag']:
                    child = ET.Element('floatvar')
                    child.set('type', str(key))
                    child.text = str(val)
                    elt.append(child)
                
        # return
        return elt

    def prettyprint(self, elt):
        """ Return printable version of element.
        """
        uglystring = ET.tostring(elt)
        prettydom = dom.parseString(uglystring)
        return prettydom.toprettyxml()
        
    def compute_match(self, name, features):
        """ Get goodness of fit for features to shape name using classifer.
        
        @param self The object pointer
        
        @param name String for shape name
        
        @param features Shape object to be matched
        
        @retval Float between 0 and 1 for degree or probability of match
        
        """
        # forward request to classifier
        return self._classifier.compute_match(self._data[name], features)