"""
Purpose:

This code returns the titles of the 10 most similar wikipedia articles against a 
given query wikipedia article title as measured by L1 distance.



Suggested use:

Copy code into an IDE that keeps variables in memory (Spyder, Jupyter, etc.). This way,
the datasets in memory and connection to Nearist server will persist while you 
iteratively search new query terms and adjust parameters.



Query term: 

The name of a wikipedia article title. Titles are capitalized, and typically 
each word of a multiword title is capitalizaed. Refer to wikipedia article titles online.

"""
################################################################################


from __future__ import division
from nearist import Client, Common
import h5py
import time
import sys
import pickle

# Wikipedia article title 
query_term = "Crocodile"

# Connection parameters
api_key = "apikey"
nearist_port = 5555
nearist_ip = "000.00.0.0.00.0"

# Filepath parameters 
path_on_nearist_server = "/nearist/Wikipedia/lsi_index_uint8.h5"
path_on_local_drive = './wiki_data/lsi_index_uint8.h5'
path_to_titles = "./wiki_data/titles_to_id.pickle"

# Establish connection
print 'Connecting to Nearist server...'
sys.stdout.flush()
c = Client()
c.open(nearist_ip, nearist_port, api_key)
print '    Connection successful.\n'

# Set this flag to 'True' after the first run of this script so that you don't
# have to load anything again.
loaded = False

if not loaded:
    print 'Loading remote dataset...'
    sys.stdout.flush()

    # Load dataset in Nearist server
    t0 = time.time()
    c.load_dataset_file(file_name=path_on_nearist_server, dataset_name='lsi')

    print '    Done (%.0f sec)\n' % (time.time() - t0)

    print 'Loading local dataset files...'
    sys.stdout.flush()
    
    t0 = time.time()
    
    # Load article titles
    title_to_id = pickle.load( open( path_to_titles, "rb" ) )
    id_to_title = dict(zip(title_to_id.values(), title_to_id.keys()))
    
    # Load data locally to query against Nearist server
    h5f = h5py.File(path_on_local_drive, 'r')
    vecs = h5f['lsi'][:]

    print '    Done (%.0f sec)\n' % (time.time() - t0)
    sys.stdout.flush()


# Set server parameters
c.set_distance_mode(Common.DistanceMode.L1)
c.set_query_mode(Common.QueryMode.KNN_A)

# Set k = 11 to get 10 results since the top result will always be the query
# article itself.
c.set_read_count(11)


def main(query_term):
    # Given a title, retrieve the associated vector and query Nearist server
    query_id = title_to_id[query_term]
    query_vec = list(vecs[query_id])

    print 'Finding most similar articles to "%s"...' % query_term
    
    # We'll measure the latency, with and without internet overhead.
    c.reset_timer()
    t0 = time.time()
    
    result = c.query(query_vec)

    # Get the total elapsed time (including internet overhead in ms)
    wall_time = (time.time() - t0) * 1000.0

    # Get only the time spent on the appliance. Convert from nanoseconds to ms.
    hw_time = c.get_timer_value() / 1E6

    # Print ten nearest neighbors
    for i in result[1:]:
        print ('    ' + id_to_title[i['ds_id']])

    # Report the latency. 
    print '\n%20s %.0f ms' % ('Observed time:', wall_time)
    print '%20s %.0f ms' % ('Hardware time:', hw_time)
    
