%module Cconvolution2

%{
#define SWIG_FILE_WITH_INIT
#include "convolution2.h"
%}

%include "numpy.i"
%include "carrays.i"
%array_class(int, intArray);

%init %{
import_array();
%}

%apply (int INPLACE_ARRAY2[ANY][ANY]) {(int src[600][800])};
void CMultiThreshold(int src[600][800]);
void U(int *ValueList, int size);

%include "convolution2.h"

