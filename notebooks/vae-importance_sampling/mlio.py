# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

"""
Module ``misc.io`` includes useful functions for loading and saving
datasets, result tables or objects in general.

This module contains the following functions:

* ``load_from_file``:        Loads a dataset from a file without allocating memory for it.
* ``load_from_files``:       Loads a dataset from a list of files without allocating memory for them.
* ``ascii_load``:            Reads an ASCII file and returns its data and metadata.
* ``libsvm_load``:           Reads a LIBSVM file and returns its data and metadata.
* ``libsvm_load_line``:      Converts a line from a LIBSVM file in an example.
* ``save``:                  Saves an object into a file.
* ``load``:                  Loads an object from a file.
* ``gsave``:                 Saves an object into a gzipped file
* ``gload``:                 Loads an object from a gzipped file

and the following classes:

* ASCIIResultTable:     Object that loads an ASCII table and implements many useful operations.
* IteratorWithFields:   Iterator which separates the rows of a NumPy array into fields.
* MemoryDataset:        Iterator over some data put in memory as a NumPy array.
* FileDataset:          Iterator over a file whose lines are converted in examples.    
* FilesDataset:         Iterator over list of files whose content is converted in examples.    

"""

import pickle, os
import numpy as np
from gzip import GzipFile as gfile


### Some helper classes ###

class ASCIIResult():
    """
    Object representing a line in an ASCIIResultTable.
    """

    def __init__(self,values,fields):
        if len(values) != len(fields):
            raise ValueError('values and fields should be of the same size')

        self.values = values
        self.fields = fields

    def __getitem__(self,id):
        if isinstance(id,str):
            if id not in self.fields:
                raise TypeError('index %s is not a valid field' % id)
            if self.fields.count(id) > 1:
                raise ValueError('index %s is ambiguous: many fields have this name' % id)
            return self.values[self.fields.index(id)]
        else:
            return self.values[id]

    def __iter__(self):
        for val in self.values:
            yield val

    def __len__(self):
        return len(self.values)

    def __str__(self):
        return self.values.__str__()
    

class ASCIIResultTable():
    """
    Object that loads an ASCII table and implements many useful operations.

    The first row of in the ASCII table's file is assumed to be a header
    providing names for each field of the table. The remaining rows correspond
    to the results. Each field (column) of the table must be separated by 
    character ``separator`` (default is ``'\t'``). 

    If the file doesn't contain a first line header, the list of field names
    can be explicitly given using option ``fields``).
    """

    def __init__(self,file,separator='\t',fields=None):

        self.file = file
        self.separator = separator

        stream = open(os.path.expanduser(self.file))

        self.all_results = [ line.rstrip('\n').split(self.separator) for line in stream]
        if fields is None:
            self.fields = self.all_results[0]
            idx = 1
        else:
            self.fields = fields
            idx = 0
        self.all_results = [ ASCIIResult(result,self.fields) for result in self.all_results[idx:] ]
        def filter_func(item):
            return True

        self.filter_func = filter_func
        self.results = filter(self.filter_func,self.all_results)

    def sort(self,field,numerical=False):
        """
        Sorts the rows of the table based on the value of the field at
        position ``field``. ``field`` can also be a string field name. 
        If ``numerical`` is True, then the
        numerical values are used for sorting, otherwise sorting is
        based on the string value.
        """

        if numerical:
            def key(a):
                return float(a[field])
        else:
            def key(a):
                return a[field]

        self.all_results.sort(key=key)
        self.results = filter(self.filter_func,self.all_results)

    def filter(self,filter_func):
        """
        Filters the rows of the table by keeping those for which
        the output of function ``filter_func`` is True. This will
        overwrite any previous filtering function (i.e. filtering
        functions are not sequentially composed).
        """
        self.filter_func = filter_func
        self.results = filter(self.filter_func,self.all_results)

    def __getitem__(self,row_id):
        if isinstance(row_id,tuple):
            if len(row_id) != 2:
                raise TypeError('indices must be integers or pairs')
            return self.results[row_id[0]][row_id[1]]
        else:
            return self.results[row_id]

    def __iter__(self):
        for result in self.results:
            yield result

    def __len__(self):
        return len(self.results)

    def __str__(self):
        # figure out the length of all elements in the table
        all_lengths = [ [ len(elem) for elem in elements ] for elements in [self.fields] + self.results]

        # figure max length in each column
        max_lengths = [ max([ lengths[i] for lengths in all_lengths ]) for i in range(len(all_lengths[0]))]
        
        def format_line(line,max_lengths):
            tokens = line.split(self.separator)
            tokens = [' '*(max_lengths[i]-len(tokens[i]))+tokens[i] for i in range(len(tokens))]
            return '  '.join(tokens)

        ret = format_line(self.separator.join(self.fields),max_lengths)
        for result in self.results:
            ret += '\n' + format_line(self.separator.join(result),max_lengths)

        return ret


class IteratorWithFields():
    """
    An iterator over the rows of a NumPy array, which separates each row into fields (segments)

    This class helps avoiding the creation of a list of arrays.
    The fields are defined by a list of pairs (beg,end), such that 
    data[:,beg:end] is a field.
    """

    def __init__(self,data,fields):
        self.data = data
        self.fields = fields

    def __iter__(self):
        for r in self.data:
            yield [ (r[beg] if beg+1==end else r[beg:end]) for (beg,end) in self.fields ]


class MemoryDataset():
    """
    An iterator over some data, but that puts the content 
    of the data in memory in NumPy arrays.

    Option ``'field_shapes'`` is a list of tuples, corresponding
    to the shape of each fields.

    Option ``dtypes`` determines the type of each field (float, int, etc.).

    Optionally, the length of the dataset can also be
    provided. If not, it will be figured out automatically.
    """

    def __init__(self,data,field_shapes,dtypes,length=None):
        self.data = data
        self.field_shapes = field_shapes
        self.n_fields = len(field_shapes)
        self.mem_data = []
        if length == None:
            # Figure out length
            length = 0
            for example in data:
                length += 1
        self.length = length
        for i in range(self.n_fields):
            sh = field_shapes[i]
            if sh == (1,):
                mem_shape = (length,) # Special case of non-array fields. This will 
                                      # ensure that a non-array field is yielded
            else:
                mem_shape = (length,)+sh
            self.mem_data += [np.zeros(mem_shape,dtype=dtypes[i])]

        # Put data in memory
        t = 0
        if self.n_fields == 1:
            for example in data:
                self.mem_data[0][t] = example
                t+=1
        else:
            for example in data:
                for i in range(self.n_fields):
                    self.mem_data[i][t] = example[i]
                t+=1

    def __iter__(self):
        if self.n_fields == 1:
            for example in self.mem_data[0]:
                yield example
        else:
            for t in range(self.length):
                yield tuple( m[t] for m in self.mem_data )


class FileDataset():
    """
    An iterator over a dataset file, which converts each
    line of the file into an example.

    The option ``'load_line'`` is a function which, given 
    a string (a line in the file) outputs an example.
    """

    def __init__(self,filename,load_line):
        self.filename = filename
        self.load_line = load_line

    def __iter__(self):
        stream = open(os.path.expanduser(self.filename))
        for line in stream:
            yield self.load_line(line)
        stream.close()

class FilesDataset():
    """
    An iterator over dataset files, wich converts each
    file of the list into an example.

    The option ``'load_file'`` is a function which, given 
    a string (the content of a file) outputs an example.
    """

    def __init__(self, filenames, load_file):
        self.filenames = filenames
        self.load_file = load_file

    def __iter__(self):
        for filename in self.filenames:
            stream = open(os.path.expanduser(filename))
            string = stream.read()
            stream.close()
            yield self.load_file(string)

    def __len__(self):
        return len(self.filenames)


### For loading large datasets which don't fit in memory ###

def load_line_default(line):
    return np.array([float(i) for i in line.split()]) # Converts each element to a float

def load_from_file(filename,load_line=load_line_default):
    """
    Loads a dataset from a file, without loading it in memory.

    It returns an iterator over the examples from that fine. This is based
    on class ``FileDataset``.
    """
    return FileDataset(filename,load_line)

def load_file_default(file):
    return np.array([float(i) for i in line.split()]) # Converts each element to a float

def load_from_files(filenames, load_file=load_file_default):
    """
    Loads a dataset from a list of files, without loading them in memory.

    It returns an iterator over the examples from these fines. This is based
    on class ``FilesDataset``.
    """
    return FilesDataset(filenames,load_file)
    

# Functions to load datasets in different common formats.
# Those functions output a pair (data,metadata), where 
# metadata is a dictionary.

### ASCII format ###

def ascii_load(filename, convert_input=float, last_column_is_target = False, convert_target=float):
    """
    Reads an ASCII file and returns its data and metadata.

    Data can either be a simple NumPy array (matrix), or an iterator
    over (numpy array,target) pairs if the last column of the ASCII
    file is to be considered a target.

    Options ``'convert_input'`` and ``'convert_target'`` are functions
    which must convert an element of the ASCII file from the string
    format to the desired format.

    **Defined metadata:**

    * ``'input_size'``

    """

    f = open(os.path.expanduser(filename))
    lines = f.readlines()

    if last_column_is_target == 0:
        data = np.array([ [ convert_input(i) for i in line.split() ] for line in lines])
        return (data,{'input_size':data.shape[1]})
    else:
        data = np.array([ [ convert_input(i) for i in line.split()[:-1] ] + [convert_target(line.split()[-1])] for line in lines])
        return (IteratorWithFields(data,[(0,data.shape[1]-1),(data.shape[1]-1,data.shape[1])]),
                {'input_size':data.shape[1]-1})
    f.close()

### LIBSVM format ###

def libsvm_load_line(line,convert_non_digit_features=float,convert_target=str,sparse=False,input_size=-1):
    """
    Converts a line (string) of a LIBSVM file into an example (list).

    This function is used by ``libsvm_load()``.
    If ``sparse`` is False, option ``'input_size'`` is used to determine the size 
    of the returned 1D array  (it must be big enough to fit all features).
    """
    if line.find('#') >= 0:
        line = line[:line.find('#')] # Remove comments

    line = line.rstrip() # Must not remove whitespace on the left, for multi-label datasets
                         # where an empty labels means all labels are 0.
    tokens = line.split(' ')

    # Always keep the first token (target)
    # but remove other empty tokens (happens when 
    # features are separated by more than one space
    def non_empty(x):
        return len(x) > 0
    tokens = tokens[0:1] + filter(non_empty,tokens[1:]) 
        
    # Remove indices < 1
    n_removed = 0
    n_feat = 0
    for token,i in zip(tokens, range(len(tokens))):
        if token.find(':') >= 0:
            if token[:token.find(':')].isdigit():
                if int(token[:token.find(':')]) < 1: # Removing feature ids < 1
                    del tokens[i-n_removed]
                    n_removed += 1
                else:
                    n_feat += 1
        
    if sparse:
        inputs = np.zeros((n_feat))
        indices = np.zeros((n_feat),dtype='int')
    else:
        input = np.zeros((input_size))
    extra = []

    i = 0
    for token in tokens[1:]:
        id_str,input_str = token.split(':')
        if id_str.isdigit():
            if sparse:
                indices[i] = int(id_str)
                inputs[i] = float(input_str)
            else:
                input[int(id_str)-1] = float(input_str)
            i += 1
        else:
            extra += [convert_non_digit_features(id_str,input_str)]
            
    if sparse:
        example = [(inputs, indices), convert_target(tokens[0])]
    else:
        example = [input,convert_target(tokens[0])]
    if extra:
        example += extra
    return example

def libsvm_load(filename,convert_non_digit_features=float,convert_target=str,sparse=False,input_size=None,compute_targets_metadata=True):
    """
    Reads a LIBSVM file and returns the list of all examples (data)
    and metadata information.

    In general, each example in the list is a two items list ``[input, target]`` where

    * if ``sparse`` is True, ``input`` is a pair (values, indices) of two vectors 
      (vector of values and of indices). Indices start at 1;
    * if ``sparse`` is False, ``input`` is a 1D array such that its elements
      at the positions given by indices-1 are set to the associated values, and the
      other elemnents are 0;
    * ``target`` is a string corresponding to the target to predict.

    If a ``feature:value`` pair in the file is such that ``feature`` is not an integer, 
    ``value`` will be converted to the desired format using option
    ``convert_non_digit_features``. This option must be a callable function
    taking 2 string arguments, and will be called as follows: ::

       output = convert_non_digit_features(feature_str,value_str)

    where ``feature_str`` and ``value_str`` are ``feature`` and ``value`` in string format.
    Its output will be appended to the list of the given example.

    The input_size can be given by the user. Otherwise, will try to figure
    it out from the file (won't work if the file format is sparse and some of the
    last features are all 0!).

    The metadata 'targets' (i.e. the set of instantiated targets) will be computed
    by default, but it can be ignored using option `compute_targets_metadata=False``.

    **Defined metadata:**

    * ``'targets'`` (if ``compute_targets_metadata`` is True)
    * ``'input_size'``

    """

    stream = open(os.path.expanduser(filename))
    data = []
    metadata = {}
    if compute_targets_metadata:
        targets = set()

    if input_size is None:
        given_input_size = None
        input_size = 0
    else:
        given_input_size = input_size

    for line in stream:
        example = libsvm_load_line(line,convert_non_digit_features,convert_target,True)
        max_non_zero_feature = max(example[0][1])
        if (given_input_size is None) and (max_non_zero_feature > input_size):
            input_size = max_non_zero_feature
        if compute_targets_metadata:
            targets.add(example[1])
        # If not sparse, first pass through libsvm file just 
        # figures out the input_size and targets
        if sparse:
            data += [example]
    stream.close()

    if not sparse:
        # Now that we know the input_size, we can load the data
        stream = open(os.path.expanduser(filename))
        for line in stream:
            example = libsvm_load_line(line,convert_non_digit_features,convert_target,False,input_size)
            data += [example]
        stream.close()
        
    if compute_targets_metadata:
        metadata['targets'] = targets
    metadata['input_size'] = input_size
    return data, metadata

### Generic save/load functions, using pickle ###

def save(p, filename):
    """
    Pickles object ``p`` and saves it to file ``filename``.
    """
    f=file(filename,'wb')
    pickle.dump(p,f,pickle.HIGHEST_PROTOCOL) 
    f.close()

def load(filename): 
    """
    Loads pickled object in file ``filename``.
    """
    f=file(filename,'rb')
    y=pickle.load(f)
    f.close()
    return y

def gsave(p, filename):
    """
    Same as ``save(p,filname)``, but saves into a gzipped file.
    """
    f=gfile(filename,'wb')
    pickle.dump(p,f,pickle.HIGHEST_PROTOCOL) 
    f.close()

def gload(filename):
    """
    Same as ``load(filname)``, but loads from a gzipped file.
    """    
    f=gfile(filename,'rb')
    y=pickle.load(f)
    f.close()
    return y

