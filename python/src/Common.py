import struct
import binascii

from enum import IntEnum

class DistanceMode(IntEnum):
    """
     Possible distance metric configurations.

     The distance metric can not be changed from software, but the software
     needs to be configured for the hardware's distance metric.
    """
    NO_DISTANCE_MODE = 0xFFFF
    """
    L1 distance, a.k.a. the Manhattan or Taxi Cab distance.

    The L1 distance is calculated as the sum of absolute differences.
    It's used as a more computationally efficient alternative to the L2
    distance. It is also a suitable alternative to the Cosine distance
    provided that the vectors have been normalized first.
    """
    L1 = 0x0000
    LMAX = 0x0001
    """
    Hamming distance.

    The Hamming distance is calculated by comparing each component of
    the vectors and counting those which are not equal.
    The components could be single bits or integers.
    In the case of single bit components, the Hamming distance is
    equivalent to performing XOR and counting the number of bits.
    """
    HAMMING = 0x0002
    """
    Bitwise AND similarity.

    This similarity metric only applies for 1-bit components.
    It is calculating by performing AND and counting the resulting
    number of bits.
    """
    BIT_AND = 0x0003
    BIT_OR = 0x0004
    """
    Jaccard (a.k.a. Tanimoto) similarity.

    The Jaccard similarity metric only applies for 1-bit components.
    It is calculated by dividing the size of the intersection between
    two bit sets by the size of their union.

    For bit-vectors, the size of the intersection is determined by
    performing bitwise AND and counting the number of bits set. The size
    of the union is determined by performing bitwise OR and counting the
    number of bits set.
    """
    JACCARD = 0x0005


class QueryMode(IntEnum):
    """
    Possible query mode configurations.
    """
    NO_QUERY_MODE = 0xFFFF
    ALL = 0x0000
    """
    k-NN ascending, lowest values returned first.

    This mode is typically used with distance metrics (as opposed to
    similarity metrics) since lowest distance implies the best match.

    See also Client.set_read_count
    """
    KNN_A = 0x0001
    """
    k-NN descending, highest values returned first.

    This mode is typically used with similarity metrics (as opposed to
    distance metrics) since highest similarity implies the best match.

    See also set_read_count
    """
    KNN_D = 0x0002
    """
    Return only results with a value greater than the set threshold.

    Results with a value less than or equal to the threshold are *not*
    returned.

    This mode is typically used with similarity metrics (as opposed to
    distance metrics) since similarities *above* a threshold implies
    the best match.

    See also Client.set_threshold
    """
    GT = 0x0003
    """
    Return only results with a value less than the set threshold.

    Results with a value greater than or equal to the threshold are
    *not* returned.

    This mode is typically used with distance metrics (as opposed to
    similarity metrics) since distances *below* a threshold implies
    the best match.

    See also Client.set_threshold
    """
    LT = 0x0004
    EQ = 0x0005
    RANGE = 0x0006


class Command(IntEnum):
    RESET = 0x00
    DISTANCE_MODE = 0x01
    QUERY_MODE = 0x02
    READ_COUNT = 0x03
    THRESHOLD = 0x04
    DS_LOAD = 0x05
    QUERY = 0x06
    RESET_TIMER = 0x10
    GET_TIMER = 0x11


class Status(IntEnum):
    SUCCESS = 0x00
    INVALID_SEQUENCE = 0x01
    INVALID_ARGUMENT = 0x02
    INVALID_PACKET = 0x03
    NOT_SUPPORTED = 0x04
    INVALID_COMMAND = 0x05
    INVALID_DATA = 0x06
    TIMEOUT = 0x07
    INVALID_CHECKSUM = 0x08
    INVALID_API_KEY = 0x09
    DATASET_FILE_NOT_FOUND = 0x20
    DATASET_NOT_FOUND = 0x21
    DATASET_SIZE_NOT_SUPPORTED = 0x22
    QUERY_SIZE_NOT_SUPPORTED = 0x23
    DISTANCE_MODE_NOT_SUPPORTED = 0x24
    QUERY_MODE_NOT_SUPPORTED = 0x25
    READ_COUNT_NOT_SUPPORTED = 0x26
    UNKNOWN_ERROR = 0xFF


class Request:
    def __init__(self, api_key, command, attribute_0=0, attribute_1=0, body_length=0, body=None):
        # Pad the API key out to 8 characters, and only take 8 characters.
        self.api_key = api_key.ljust(8)[0:8]
        self.command = command
        self.attribute_0 = attribute_0
        self.attribute_1 = attribute_1
        self.body_length = body_length
        self.body = body

    def pack(self):
        """
        Returns the binary representation of this request (as a string).
        
        The header structure is as follows:
           Command      4 bytes
           Reserved     4 bytes
           Attribute 0  8 bytes
           Attribute 1  8 bytes
           Body length  8 bytes
           API Key      8 bytes (8 characters)
           Checksum     4 bytes
        
        The header is followed by the body of the request, if present.
        """

        # Pack the message header.        
        # 'L' is unsigned long (32-bit) and 'Q' is unsigned long long (64-bit)
        # 'LLQQQ' = 2*4 + 3*8 = 32 bytes
        buf = struct.pack("=LLQQQ", self.command, 0, self.attribute_0, self.attribute_1, self.body_length)

        # Add the API key to the header (it's 8 characters, 8 bytes).
        buf += format(self.api_key).encode()
        
        # Add a checksum of the header fields.
        buf += struct.pack("=L", binascii.crc32(buf) & 0xFFFFFFFF)

        # If this request includes a body...
        if self.body_length > 0 and self.body is not None:
            body = ""
            
            # For each vector...
            for vector in self.body:
                # TODO - This doesn't support numpy arrays.
                if isinstance(vector, list):
                    # Add each component of the vector to the request.
                    # TODO - B is uint8, so this is currently hardcoded to 
                    #        components of that size.
                    if(isinstance(body,str)):
                        body = format(body).encode() # this may not work for batch queries this needs to be done before entering the loop
                    for comp in vector:
                        body += struct.pack("B", comp)
                else:
                    if isinstance(vector, str):
                        body += vector
                    else:
                        body += struct.pack("B", vector)
            if isinstance(body, str):
                body = format(body).encode()
            # Append the body and a checksum of it to the end of the buffer.
            buf = buf + body + struct.pack("=L", binascii.crc32(body) & 0xFFFFFFFF) #this line is correct

        return buf


class Response:
    """
    Class representing a response received from the appliance.
    
    The response header is 36 bytes:
        (4) command
        (4) status
        (8) attribute_0
        (8) attribute_1
        (8) body_length
        (4) header checksum
    
    The body is received over the socket separately from the header.
    First we receive 36 bytes to receive the header, then this tells us
    how many bytes to receive for the body.
    """
    def __init__(self, command=Command.RESET, status=Status.SUCCESS, attribute_0=0, attribute_1=0, body_length=0,
                 body=None):
        self.command = command
        self.status = status
        self.attribute_0 = attribute_0
        self.attribute_1 = attribute_1
        self.body_length = body_length
        self.checksum = 0
        self.body = body
        self.body_checksum = 0

    def unpack_header(self, buffer):
        (self.command, self.status, self.attribute_0, self.attribute_1, self.body_length, self.checksum) = \
            struct.unpack_from("=LLQQQL", buffer, 0)
