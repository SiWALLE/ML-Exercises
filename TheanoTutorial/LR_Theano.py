# author 'liangsibiao'

# initialize with 0 the weights W as a matrix of shape(n_in,n_out)
self.W = theano.shared(value = numpy.zeros((n_in,n_out),\
	dtype = theano.configfoatX),name = 'W', borrow = True)
# initialize the biases b as a vector of n_out 0s
self.b = theano.shared(value=numpy.zeros((n_out,),\
	dtype=theano.config.floatX),name = 'b', borrow = True)
