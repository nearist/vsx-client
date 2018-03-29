import numpy as np

class Transforms:
    """
    This class implements common vector transformations applied to vectors
    before using them with the Nearist hardware.

    This includes L1 and L2 normalization and various floating point to integer
    mapping techniques.
    """

    def __init__(self, datatype):
        """"
        Parameters:
            datatype - Map the floats to integers in the range of datatype.
                E.g., for np.uint8, map values in the range [0,255]
        """
        self.int_size = datatype

        # Hardcoded maximum integer value corresponding to different datatypes
        if datatype is np.uint8:
            self.max_int = 255
        elif datatype is np.uint16:
            self.max_int = 65535
        elif datatype is np.uint32:
            self.max_int = 4294967295
        else:
            print "Please specify a datatype: [np.uint8, np.uint16, np.uint32]"


    @classmethod
    def l2_normalize(cls, X):
        """
        Normalize the row vectors in X with the L2 norm.

        Parameters:
            X - A numpy matrix containing row vectors.

        Returns:
            X with normalized row vectors.
        """
        l2norms = np.linalg.norm(X, axis=1, ord=2)
        l2norms = l2norms.reshape(len(l2norms), 1)

        return X / l2norms

    @classmethod
    def l1_normalize(cls, X):
        """
        Normalize the row vectors in X with the L1 norm.

        Parameters:
            X - A numpy matrix containing row vectors.

        Returns:
            X with normalized row vectors.
        """
        l1norms = np.linalg.norm(X, axis=1, ord=1)
        l1norms = l1norms.reshape(len(l1norms), 1)

        return X / l1norms



    def learn_logistic(self, X, num_std=8):
        """
        Learn a logistic mapping function for integer conversion.

        To learn the mapping function, we need to identify the input saturation
        point, which we call "x_sat". That is, at -x_sat, the output is 0, and
        at +x_sat, the output is max_int.

        x_sat is calculated as a multiple of the standard deviation of the
        values in 'X'. 'num_std' is the number of multiples.

        So x_sat = num_std * numpy.std(X)

        Parameters:
            X     - Matrix of floating point values to learn the mapping
                    parameters from.
            num_std - Number of standard deviations to saturate at.

        """
        
        # TODO: automatically zero-center the data
        
        # Check that values are zero-centered
        print "The logistic function requires that your data is zero-centered. \
        Please check that your data is zero-centered before continuing."        

        # Calculate the saturation point.
        self.x_sat = num_std * np.std(X)

        # Calculate the coefficient 'alpha' to use in the mapping function
        # based on the  saturation point.
        # The logistic mapping function is: 1 / (1 + exp(-alpha * X))
        self.alpha = -1/self.x_sat * np.log(0.5/(self.max_int - 0.5));

    def apply_logistic(self, X):
        """
        Use the logistic function to map floating point values to integers.

        Parameters:
            X     - Matrix of floating point values to be mapped to integers.
        """

        # Apply the logistic function to the data.
        # The output of the logistic function is bound by [0, 1.0].
        X = 1 / (1 + np.exp(-self.alpha * X))

        # Map the values from 0 - 1 to the integer range.
        X = np.rint(X * self.max_int)

        return X.astype(int)


    def learn_logarithmic(self, X):
        """
        A logarthmic scale mapping function for conversion from float to integers
        in range [0,self.max_int].

        This function learns and stores the log maximum value from matrix X and
        stores it as self.logarthmic_max.

        Parameters:
            X     - Matrix of floating point values to learn the maximum value
                    from.
        """

        # Check that all the values are positive
        assert data.all() >= 0, "The logarithmic transform requires that your data is \
        greater than or equal to zero. Please check that your data is greater than or \
        equal to zero before continuing."

        # np.max finds the maximum of all values in the dataset.
        # 1 is added in order to avoid taking the log of 0, which is undefined
        self.logarithmic_max = np.log(np.amax(X)+1)

    def apply_logarithmic(self, X):
        """
        Use the previously learned logarithmic_max as denominator in our scaling
        function to convert float values to integers.

        Parameters:
            X     - Matrix of floating point values to be converted to integers.

        Returns:
                  - Matrix of integer values in range [0,self.max_int].

        """
        # 1 is added in order to avoid taking the log of 0, which is undefined
        return np.round(np.log(X+1) / self.logarithmic_max * self.max_int).astype(self.int_size)
