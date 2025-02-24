import numpy as np
import re

# data_out = np.arange(10, dtype=np.uint16)
data_out = [np.arange(10, dtype=np.uint16), np.arange(5, dtype=np.uint16)]

# Temporarily adjust print options
with np.printoptions(threshold=np.inf):
    if type(data_out) != list:
        data_out = [data_out]
    data_out_str = str(data_out)

print(data_out_str[1:-1])

value_strings = re.findall(r'\[(.*?)\]', data_out_str[1:-1])
dtypes = re.findall(r'dtype=(.*?)\)', data_out_str[1:-1])
print(value_strings)
print(dtypes)

data_out_parsed = [np.fromstring(string=value_string, dtype=dtype, sep=',') for value_string, dtype in zip(value_strings, dtypes)]

for array, array_parsed in zip(data_out, data_out_parsed):
    assert np.array_equal(array, array_parsed)