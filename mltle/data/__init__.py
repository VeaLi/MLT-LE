from mltle.data import maps

try:
	from mltle.data import graphs
except:
    print('Failed to load RDkit. `mltle.data.graphs` is not available')
    print(e)