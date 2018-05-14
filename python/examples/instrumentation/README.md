# Instrumentation Examples
The examples in this directory are intended to demonstrate the instrumentation available in the Client API for measuring appliance time versus "wall time" (the total time experienced by the user, including internet overhead).

`result_count_latency.ipynb` - This notebook measures the additional latency added by each result returned by the hardware (it turns out to be about 20us per result).