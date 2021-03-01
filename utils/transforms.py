import numpy as np


in_lower = -1.5
in_upper = 1.5
out_lower = -127
out_upper = 128


class Float16ToInt8(object):
    def __call__(self, item):
        readings, label = item
        new_readings = (readings - in_lower) / (in_upper - in_lower)
        new_readings = new_readings * (out_upper - out_lower) + out_lower
        new_readings = new_readings.astype(np.int8).astype(np.float32)
        # for i in range(len(readings)):
        #     print(readings[i][0], new_readings[i][0])
        return new_readings, label