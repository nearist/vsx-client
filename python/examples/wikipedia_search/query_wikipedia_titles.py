"""

To run:
- python query_wikipedia_titles.py Query

Parameters
    ----------
    Query : The name of a wikipedia article title. Titles are capitalized, and typically 
    each word of a multiword title is capitalizaed.
        
Returns
    -------
	The titles of the ten most similar wikipedia articles, as measured by L1 distance.
	

"""
################################################################################


from __future__ import division
from nearist import Client, Common
import h5py
import time
import sys
import numpy as np
import pickle
import string

# Connection parameters
api_key = "apikey"
nearist_port = 5555
nearist_ip = "000.00.0.0.00.0"

# Filepath parameters 
path_on_nearist_server = "/nearist/Wikipedia/lsi_index_uint8.h5"
path_to_titles = "wiki_data/titles_to_id.pickle"
path_on_local_drive = './wiki_data/lsi_index_uint8.h5'

# Establish connection
print 'Connecting to Nearist server...'
sys.stdout.flush()
c = Client()
c.open(nearist_ip, nearist_port, api_key)
print 'Connection successful. Loading datasets...'

# Clear server parameters
c.reset()

# Load dataset in Nearist server
c.load_dataset_file(file_name=path_on_nearist_server, dataset_name='lsi')

# Set server parameters
c.set_distance_mode(Common.DistanceMode.L1)
c.set_query_mode(Common.QueryMode.KNN_A)
c.set_read_count(11)

# Load article titles
title_to_id = pickle.load( open( path_to_titles, "rb" ) )
id_to_title = dict(zip(title_to_id.values(), title_to_id.keys()))

# Load data locally to query against Nearist server
h5f = h5py.File(path_on_local_drive, 'r')
vecs = h5f['lsi'][:]


def main(query_term):
	# Given a title, retrieve the associated vector and query Nearist server
	query_id = title_to_id[query_term]
	query_vec = list(vecs[query_id])
	result = c.query(query_vec)

	# Print ten nearest neighbors
	for i in result[1:]:
		print (id_to_title[i['ds_id']])

    
if __name__ == "__main__":
    query = string.join(sys.argv[1:])
    main(query)