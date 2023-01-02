import numpy as np


def gen_2000_600(fig):
    t1 = 1.99 * np.random.random((2000, 1)) + 0.01
    t2 = 1.99 * np.random.random((2000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (2000, 1))
    e2 = np.random.normal(0, 0.1, (2000, 1))
    e3 = np.random.normal(0, 0.1, (2000, 1))
    e4 = np.random.normal(0, 0.1, (2000, 1))
    e5 = np.random.normal(0, 0.1, (2000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    output2f[0: 1000, :] = output2[0: 1000, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # anomaly detection changes
    output2f[1000: 1300, :] = output2[1000: 1300, :] + fault
    output2f[1300: 1600, :] = output2[1300: 1600, :]
    output2f[1600: 2000, :] = output2[1600: 2000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault


def gen_3000_600(fig):
    t1 = 1.99 * np.random.random((3000, 1)) + 0.01
    t2 = 1.99 * np.random.random((3000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (3000, 1))
    e2 = np.random.normal(0, 0.1, (3000, 1))
    e3 = np.random.normal(0, 0.1, (3000, 1))
    e4 = np.random.normal(0, 0.1, (3000, 1))
    e5 = np.random.normal(0, 0.1, (3000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    output2f[0: 1000, :] = output2[0: 1000, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # anomaly detection changes
    output2f[1000: 1300, :] = output2[1000: 1300, :] + fault
    output2f[1300: 1600, :] = output2[1300: 1600, :]
    output2f[1600: 2000, :] = output2[1600: 2000, :] + fault

    output2f[2000: 3000, :] = output2[2000: 3000, :]

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free.T, output_free.T, input_fault.T, output_fault.T


def gen_12000_600(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    output2f[0: 1000, :] = output2[0: 1000, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # anomaly detection changes
    output2f[1000: 1300, :] = output2[1000: 1300, :] + fault
    output2f[1300: 1600, :] = output2[1300: 1600, :]
    output2f[1600: 2000, :] = output2[1600: 2000, :] + fault

    output2f[2000: 12000, :] = output2[2000: 12000, :]

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault


def gen_12000_600_last(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    output2f[0: 11000, :] = output2[0: 11000, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # anomaly detection changes
    output2f[11000: 11300, :] = output2[11000: 11300, :] + fault
    output2f[11300: 11600, :] = output2[11300: 11600, :] # + fault # temp
    output2f[11600: 12000, :] = output2[11600: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    print('input shape: ' + str(input_free.shape))
    print('output shape: ' + str(output_free.shape))

    return input_free.T, output_free.T, input_fault.T, output_fault.T

def gen_50collective(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    # also, slicing in numpy - https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    output2f[0: 10950, :] = output2[0: 10950, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # collective anomaly detection changes

    output2f[10950: 11000, :] = output2[10950: 11000, :] + fault
    output2f[11000: 11050, :] = output2[11000: 11050, :]
    output2f[11050: 11100, :] = output2[11050: 11100, :] + fault
    output2f[11100: 11150, :] = output2[11100: 11150, :]
    output2f[11150: 11200, :] = output2[11150: 11200, :] + fault
    output2f[11200: 11250, :] = output2[11200: 11250, :]
    output2f[11250: 11300, :] = output2[11250: 11300, :] + fault
    output2f[11300: 11350, :] = output2[11300: 11350, :]
    output2f[11350: 11400, :] = output2[11350: 11400, :] + fault
    output2f[11400: 11450, :] = output2[11400: 11450, :]
    output2f[11450: 11500, :] = output2[11450: 11500, :] + fault
    output2f[11500: 11550, :] = output2[11500: 11550, :]
    output2f[11550: 11600, :] = output2[11550: 11600, :] + fault
    output2f[11600: 11650, :] = output2[11600: 11650, :]
    output2f[11650: 11700, :] = output2[11650: 11700, :] + fault
    output2f[11700: 11750, :] = output2[11700: 11750, :]
    output2f[11750: 11800, :] = output2[11750: 11800, :] + fault
    output2f[11800: 11850, :] = output2[11800: 11850, :]
    output2f[11850: 11900, :] = output2[11850: 11900, :] + fault
    output2f[11900: 11950, :] = output2[11900: 11950, :]
    output2f[11950: 12000, :] = output2[11950: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault


def gen_100collective(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    # also, slicing in numpy - https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    output2f[0: 10900, :] = output2[0: 10900, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # collective anomaly detection changes
    output2f[10900: 11000, :] = output2[10900: 11000, :] + fault
    output2f[11000: 11100, :] = output2[11000: 11100, :]
    output2f[11100: 11200, :] = output2[11100: 11200, :] + fault
    output2f[11200: 11300, :] = output2[11200: 11300, :]
    output2f[11300: 11400, :] = output2[11300: 11400, :] + fault
    output2f[11400: 11500, :] = output2[11400: 11500, :]
    output2f[11500: 11600, :] = output2[11500: 11600, :] + fault
    output2f[11600: 11700, :] = output2[11600: 11700, :]
    output2f[11700: 11800, :] = output2[11700: 11800, :] + fault
    output2f[11800: 11900, :] = output2[11800: 11900, :]
    output2f[11900: 12000, :] = output2[11900: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault

def gen_125collective(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    # also, slicing in numpy - https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    output2f[0: 10875, :] = output2[0: 10875, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # collective anomaly detection changes
    output2f[10875: 11000, :] = output2[10875: 11000, :] + fault
    output2f[11000: 11125, :] = output2[11000: 11125, :]
    output2f[11125: 11250, :] = output2[11125: 11250, :] + fault
    output2f[11250: 11375, :] = output2[11250: 11375, :]
    output2f[11375: 11500, :] = output2[11375: 11500, :] + fault
    output2f[11500: 11625, :] = output2[11500: 11625, :]
    output2f[11625: 11750, :] = output2[11625: 11750, :] + fault
    output2f[11750: 11875, :] = output2[11750: 11875, :]
    output2f[11875: 12000, :] = output2[11875: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault

def gen_150collective(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    # also, slicing in numpy - https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    output2f[0: 10950, :] = output2[0: 10950, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # collective anomaly detection changes
    output2f[10950: 11100, :] = output2[10950: 11100, :] + fault
    output2f[11100: 11250, :] = output2[11100: 11250, :]
    output2f[11250: 11400, :] = output2[11250: 11400, :] + fault
    output2f[11400: 11550, :] = output2[11400: 11550, :]
    output2f[11550: 11700, :] = output2[11550: 11700, :] + fault
    output2f[11700: 11850, :] = output2[11700: 11850, :]
    output2f[11850: 12000, :] = output2[11850: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    return input_free, output_free, input_fault, output_fault

def gen_200collective(fig):
    t1 = 1.99 * np.random.random((12000, 1)) + 0.01
    t2 = 1.99 * np.random.random((12000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (12000, 1))
    e2 = np.random.normal(0, 0.1, (12000, 1))
    e3 = np.random.normal(0, 0.1, (12000, 1))
    e4 = np.random.normal(0, 0.1, (12000, 1))
    e5 = np.random.normal(0, 0.1, (12000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    # https://stackoverflow.com/questions/16815928/what-does-mean-on-numpy-arrays
    # also, slicing in numpy - https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
    output2f[0: 11000, :] = output2[0: 11000, :]
    # output2f[1000: 2000, :] = output2[1000: 2000, :] + fault
    # collective anomaly detection changes
    output2f[11000: 11200, :] = output2[11000: 11200, :] + fault
    output2f[11200: 11400, :] = output2[11200: 11400, :]
    output2f[11400: 11600, :] = output2[11400: 11600, :] + fault
    output2f[11600: 11800, :] = output2[11600: 11800, :]
    output2f[11800: 12000, :] = output2[11800: 12000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)

    print('input shape: ' + str(input_free.shape))
    print('output shape: ' + str(output_free.shape))

    return input_free, output_free, input_fault, output_fault

def generate_data_ramp_paper(fig):
    t1 = 1.99 * np.random.random((2000, 1)) + 0.01
    t2 = 1.99 * np.random.random((2000, 1)) + 0.01

    e1 = np.random.normal(0, 0.1, (2000, 1))
    e2 = np.random.normal(0, 0.1, (2000, 1))
    e3 = np.random.normal(0, 0.1, (2000, 1))
    e4 = np.random.normal(0, 0.1, (2000, 1))
    e5 = np.random.normal(0, 0.1, (2000, 1))

    input1 = np.sin(t1) + e1
    input2 = t1 ** 2 - 3 * t1 + 4 + e2
    input3 = t1 ** 3 + 3 * t2 + e3
    output1 = input1 ** 2 + input1 * input2 + input1 + e4
    output2 = input1 * input3 + input2 + np.sin(input3) + e5

    output1f = output1
    # 产生故障
    fault = fig
    output2f = np.zeros_like(output2)
    output2f[0: 1000, :] = output2[0: 1000, :]
    output2f[1000: 2000, :] = output2[1000: 2000, :] + fault

    input_free = np.concatenate((input1, input2, input3), 1)
    output_free = np.concatenate((output1, output2), 1)
    input_fault = input_free
    output_fault = np.concatenate((output1f, output2f), 1)
    return input_free.T, output_free.T, input_fault.T, output_fault.T    # (3 * 2000)


def generate_data_ramp(fig):
    input_free, output_free, input_fault, output_fault = gen_12000_600_last(fig)
    # input_free, output_free, input_fault, output_fault = gen_200collective(fig)
    return input_free, output_free, input_fault, output_fault    # (3 * 2000)


def generate_batch(input_free, output_free, input_fault, output_fault, batch_len):
    series_len = input_free.shape[0]
    input_free_batch = []
    output_free_batch = []
    input_fault_batch = []
    output_fault_batch = []
    for i in range(series_len - batch_len + 1):
        input_free_batch.append(input_free[i: (i + batch_len)])
        output_free_batch.append(output_free[i: (i + batch_len)])
        input_fault_batch.append(input_fault[i: (i + batch_len)])
        output_fault_batch.append(output_fault[i: (i + batch_len)])
    return np.array(input_free_batch), np.array(output_free_batch), np.array(input_fault_batch), np.array(output_fault_batch)




