import json

from Common import *
import socket
import sys
import time

class Client:
    """
    This class provides the Python interface for communicating with the Nearist appliances.

    Commands are communicated via TCP/IP to the appliance server.
    """

    def __init__(self):
        self.sock = None
        

    @staticmethod
    def __recvall(sock, length):
        # Helper function to recv 'length' bytes or return None if EOF is hit.
        data = bytearray()

        # Loop until we've received 'length' bytes.        
        while len(data) < length:

            # Receive the remaining bytes. The amount of data returned might
            # be less than 'length'.
            packet = sock.recv(length - len(data))

            # If we received 0 bytes, the connection has been closed...            
            if not packet:
                return None

            # Append the received bytes to 'data'.    
            data += packet

        # Return the 'length' bytes of received data.
        return data

    def __on_request_complete(self, request):
        """
        Receives the response from the appliance.
        """

        # Receive 36 bytes (the side of the response message header).
        buf = Client.__recvall(self.sock, 36)

        if buf is not None:

            # Unpack the response header into a Response object.
            r = Response()
            r.unpack_header(buf)

            if r.status != Status.SUCCESS:
                raise IOError("Nearist error: %s " % Status(r.status))
            else:
                results = []

                # If there is data to receive for this response...
                if r.body_length > 0:
                    # Receive the body of this response.
                    r.body = Client.__recvall(self.sock, r.body_length)
                    r.body_checksum = Client.__recvall(self.sock, 4)
                    if r.body is not None:
                        length = r.body_length >> 3
                        if length > 0:
                            body = struct.unpack_from(("=" + ("Q" * length)), r.body, 0)

                            if request.attribute_1 == 0:
                                for i in range(0, length, 2):
                                    if body[i] != 0xFFFFFFFFFFFFFFFF and body[i + 1] != 0xFFFFFFFFFFFFFFFF:
                                        results.append({'ds_id': body[i], 'distance': body[i + 1]})
                                    else:
                                        break
                            elif request.attribute_1 == 1:
                                result = []
                                for i in range(0, length, 2):
                                    if body[i] != 0xFFFFFFFFFFFFFFFF and body[i + 1] != 0xFFFFFFFFFFFFFFFF:
                                        result.append({'ds_id': body[i], 'distance': body[i + 1]})
                                    else:
                                        results.append(result)
                                        result = []
                    else:
                        raise IOError("Read error.")
                else:
                    results = r.attribute_0
        else:
            raise IOError("Read error.")

        return results

    def __request(self, request):
        # 'sendall' will send all bytes in the request before returning.        
        status = self.sock.sendall(request.pack())

        # If the request was sent successfully... 
        # ('sendall' returns None if successful.)
        if status is None:
            return self.__on_request_complete(request)

    def open(self, host, port, api_key):
        """
        Open a socket for communication with the Nearist appliance.
        
        :type host: string
        :param host: IP address of the Nearist appliance.
        
        :type port: integer
        :param port: Port number for accessing the Nearist appliance.
        
        :type api_key: string
        :param api_key: Unique user access key which is required to access the
                    appliance.

        """

        # Convert the host name and port to a 5-tuple of arguments.
        # We just need the "address family" parameter.
        address_info = socket.getaddrinfo(host, port)

        # Create a new socket (the host and port are specified in 'connect').
        self.sock = socket.socket(address_info[0][0], socket.SOCK_STREAM)

        # Connect to the host.
        self.sock.connect((host, port))

        # Store the API key        
        self.api_key = api_key

    def close(self):
        """
        Close the socket to the Nearist appliance.
        """
        self.sock.close()

    def reset(self):
        """
        Reset the Nearist hardware, clearing all stored data.
        """

        # Create and submit a reset request.
        request = Request(self.api_key, Command.RESET)
        self.__request(request)

    def reset_timer(self):
        """
        Reset board time measurement timer.
        """

        request = Request(self.api_key, Command.RESET_TIMER)
        self.__request(request)

    def get_timer_value(self):
        """
        Get board time measurement in nanoseconds.
        """

        request = Request(self.api_key, Command.GET_TIMER)
        return self.__request(request)

    def set_distance_mode(self, mode):
        """
        Set the distance metric.
        
        :type mode: Common.DistanceMode
        :param mode: Distance metric from Common.DistanceMode        

        """

        # Create and submit the distance mode request.
        request = Request(
            self.api_key,
            Command.DISTANCE_MODE,
            mode
        )
        self.__request(request)

    def set_query_mode(self, mode):
        """
        Set query mode
        
        :type mode: Common.QueryMode
        :param mode: Query mode from Common.QueryMode        

        """

        request = Request(
            self.api_key,
            Command.QUERY_MODE,
            mode
        )
        self.__request(request)

    def set_read_count(self, count):
        """
        Set query result count for KNN_D/KNN_A query mode(s)
        
        :type count: integer
        :param count: The top 'K' values in KNN 

        """

        request = Request(
            self.api_key,
            Command.READ_COUNT,
            count
        )
        self.__request(request)

    def set_threshold(self, threshold):
        """
        Set query threshold for GT, LT, or KNN query modes.
        
        In GT or KNN_D mode, the appliance will only return results whose 
        distance is greater than the threshold value.

        In LT or KNN_A mode, the appliance will only return results whose 
        distance is less than the threshold value.
        
        :type threshold: integer
        :param threshold: The threshold value.
        """
        # Construct and issue the set threshold request.
        request = Request(
            self.api_key,
            Command.THRESHOLD,
            threshold
        )
        self.__request(request)        
        
    def set_threshold_range(self, threshold_lower, threshold_upper):
        """
        Set query threshold for RANGE query mode.
        
        :type threshold_lower: integer
        :param threshold_lower: Lower threshold value.
        
        :type threshold_upper: integer
        :param threshold_upper: Upper threshold value, defaults to None.
         
        """
        request = Request(
            self.api_key,
            Command.THRESHOLD,
            threshold_lower,
            threshold_upper
        )
        self.__request(request)

            

    def ds_load(self, vectors):
        """
        Load dataset to Nearist appliance
        
        :type vectors: 
        :param vectors: List of vectors (component lists)

        """

        if isinstance(vectors, list):
            if isinstance(vectors[0], list):
                request = Request(
                    self.api_key,
                    Command.DS_LOAD,
                    attribute_0=len(vectors[0]),
                    attribute_1=0,
                    body_length=len(vectors) * (len(vectors[0])),
                    body=vectors
                )
                self.__request(request)
            else:
                raise ValueError('Invalid argument')
        else:
            raise ValueError('Invalid argument')

    def load_dataset_file(self, file_name, dataset_name):
        """
        Load local dataset to Nearist appliance
        
        :type file_name: string
        :param file_name: Local dataset file name
        
        :type dataset_name: string
        :param dataset_name: Local dataset name

        """

        root = json.dumps({"fileName": file_name, "datasetName": dataset_name})
        request = Request(
            self.api_key,
            Command.DS_LOAD,
            attribute_0=0,
            attribute_1=1,
            body_length=len(root),
            body=root
        )
        self.__request(request)

    def query(self, vectors, batch_size=128, verbose=True):
        """
        Query for single/multiple vector(s)
        
        :type vectors: list or list of lists
        :param vectors: List of components for single query / List of vectors (component lists) for multipel query

        """
        # Validate that 'vectors' is a list.
        if not isinstance(vectors, list):
            raise ValueError('Invalid argument')

        # ======== Single Query ========
        # If 'vectors' is just a single query vector...
        if not isinstance(vectors[0], list):
            # Construct the query request.
            request = Request(
                self.api_key,
                Command.QUERY,
                attribute_0=len(vectors),       # Length of a vector
                body_length=len(vectors),       # Total payload size
                body=vectors
            )
            
            # Issue the query and return the results.
            return self.__request(request)
            
        # ======== Batch Query ========    
        # Otherwise, this is a batch query.
        # We will break the queries into smaller batches and aggregate
        # the results.
        
        start = 0
        results = []
        
        # Record the start time.
        t0 = time.time()
        
        # For each mini-batch...
        while start < len(vectors):
            # Calculate the 'end' of this mini-batch.
            end = min(start + batch_size, len(vectors))

            # Select the vectors in this mini-batch.
            mini_batch = vectors[start:end]

            # Progress update.
            if verbose and not start == 0:
                # Caclulate the average throughput so far.
                queries_per_sec = ((time.time() - t0)  / start)
                
                # Estimate how much time (in minutes) is left to complete the 
                # test.
                time_est = queries_per_sec * (len(vectors) - start) / 60.0
                
                # Format the estimated time remaining into minutes.
                # If it's less than 1 minute, show <1 instead of 0.
                if time_est < 1:
                    time_est_str = '<1 min...'
                else:
                    time_est_str = '~%.0f min...'

                print '  Query %5d / %5d (%.0f%%) Time Remaining: %s' % (start, len(vectors), float(start) / len(vectors) * 100.0, time_est_str)    
                sys.stdout.flush()
            
            # Construct the query request.
            request = Request(
                self.api_key,
                Command.QUERY,
                attribute_0=len(mini_batch[0]),    # Length of a vector
                attribute_1=1,                     # TODO - Currently unused?
                body_length=len(mini_batch) * (len(mini_batch[0])), # Total matrix size
                body=mini_batch 
            )
        
            # Submit the query and wait for the results.
            mini_res = self.__request(request)

            if not len(mini_batch) == len(mini_res):
                print 'ERROR: Mini batch length %d does not match results legnth %d!' % (len(mini_batch), len(mini_res))
                sys.stdout.flush()
            
            # Accumulate the results.
            results = results + mini_res
            
            # Update the start pointer.
            start = end
            
        return results
