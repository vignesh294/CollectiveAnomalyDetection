import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.io import savemat


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
    return np.array(input_free_batch), np.array(output_free_batch), np.array(input_fault_batch), np.array(
        output_fault_batch)


def far_mdr_compute(sta, thre, index_thre):
    rate_false = (sta[:index_thre] > thre).tolist().count(True) / index_thre
    rate_mdr = (sta[index_thre:] < thre).tolist().count(True) / (sta.shape[0] - index_thre)
    return rate_false, rate_mdr

def make_labels_list_2000_600():
    free_indices = []
    free_indices.append((0, 931))
    # anomaly detection weights
    free_indices.append((1232, 1531))
    fault_indices = []
    fault_indices.append((932, 1232))
    fault_indices.append((1532, 1930))
    return free_indices, fault_indices

def make_labels_list_3000_600():
    free_indices = []
    free_indices.append((0, 931))
    # anomaly detection weights
    free_indices.append((1232, 1531))
    free_indices.append((1932, 2930))
    fault_indices = []
    fault_indices.append((932, 1232))
    fault_indices.append((1532, 1931))
    return free_indices, fault_indices

def make_labels_list_12000_600():
    free_indices = []
    free_indices.append((0, 931))
    # anomaly detection weights
    free_indices.append((1232, 1531))
    free_indices.append((1932, 11930))
    fault_indices = []
    fault_indices.append((932, 1232))
    fault_indices.append((1532, 1931))
    return free_indices, fault_indices

def make_labels_list_12000_600_last():
    free_indices = []
    free_indices.append((0, 10931))
    free_indices.append((11232, 11531))
    fault_indices = []
    fault_indices.append((10932, 11231))
    fault_indices.append((11232, 11531))
    fault_indices.append((11532, 11930))
    return free_indices, fault_indices

def make_labels_list_12000_600_last_batchlen_200():
    free_indices = []
    free_indices.append((0, 10801))
    free_indices.append((11102, 11401))
    fault_indices = []
    fault_indices.append((10802, 11101))
    fault_indices.append((11102, 11401))
    fault_indices.append((11402, 11800))
    return free_indices, fault_indices

def make_labels_list_12000_600_last_batchlen_150():
    free_indices = []
    free_indices.append((0, 10851))
    free_indices.append((11152, 11451))
    fault_indices = []
    fault_indices.append((10852, 11151))
    fault_indices.append((11152, 11451))
    fault_indices.append((11452, 11850))
    return free_indices, fault_indices

def make_labels_list_12000_600_last_batchlen_100():
    free_indices = []
    free_indices.append((0, 10901))
    free_indices.append((11202, 11501))
    fault_indices = []
    fault_indices.append((10902, 11201))
    fault_indices.append((11202, 11501))
    fault_indices.append((11502, 11900))
    return free_indices, fault_indices

def make_labels_list_12000_600_last_batchlen_125():
    free_indices = []
    free_indices.append((0, 10876))
    free_indices.append((11177, 11476))
    fault_indices = []
    fault_indices.append((10877, 11176))
    fault_indices.append((11177, 11476))
    fault_indices.append((11477, 11875))
    return free_indices, fault_indices

def make_labels_list_12000_600_last_batchlen_1():
    free_indices = []
    free_indices.append((0, 10999))
    free_indices.append((11300, 11599))
    fault_indices = []
    fault_indices.append((11000, 11299))
    fault_indices.append((11600, 11999))
    return free_indices, fault_indices

def make_labels_list_12000_600_batchlen125_50collective():
    free_indices = []
    free_indices.append((0, 10825))
    free_indices.append((10876, 10925))
    free_indices.append((10976, 11025))
    free_indices.append((11076, 11125))
    free_indices.append((11176, 11225))
    free_indices.append((11276, 11325))
    free_indices.append((11376, 11425))
    free_indices.append((11476, 11525))
    free_indices.append((11576, 11625))
    free_indices.append((11676, 11725))
    free_indices.append((11776, 11825))
    fault_indices = []
    fault_indices.append((10826, 10875))
    fault_indices.append((10926, 10975))
    fault_indices.append((11026, 11075))
    fault_indices.append((11126, 11175))
    fault_indices.append((11226, 11275))
    fault_indices.append((11326, 11375))
    fault_indices.append((11426, 11475))
    fault_indices.append((11526, 11575))
    fault_indices.append((11626, 11675))
    fault_indices.append((11726, 11775))
    fault_indices.append((11826, 11875))
    labels_array = np.zeros(10826)
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(50)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(50)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

def make_labels_list_12000_600_batchlen125_100collective():
    free_indices = []
    fault_indices = []
    labels_array = np.zeros(10776)
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

def make_labels_list_12000_600_batchlen125_100collective():
    free_indices = []
    fault_indices = []
    labels_array = np.zeros(10776)
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(100)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(100)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

def make_labels_list_12000_600_batchlen125_125collective():
    free_indices = []
    fault_indices = []
    labels_array = np.zeros(10751)
    labels_array = np.concatenate((labels_array, np.ones(125)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(125)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(125)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(125)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(125)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(125)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(125)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(125)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(125)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

def make_labels_list_12000_600_batchlen125_150collective():
    free_indices = []
    fault_indices = []
    labels_array = np.zeros(10826)
    labels_array = np.concatenate((labels_array, np.ones(150)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(150)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(150)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(150)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(150)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(150)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(150)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

def make_labels_list_12000_600_batchlen125_200collective():
    free_indices = []
    fault_indices = []
    labels_array = np.zeros(10876)
    labels_array = np.concatenate((labels_array, np.ones(200)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(200)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(200)), axis=0)  # faults
    labels_array = np.concatenate((labels_array, np.zeros(200)), axis=0)  # free
    labels_array = np.concatenate((labels_array, np.ones(200)), axis=0)  # faults
    return free_indices, fault_indices, labels_array

# get the true_pos, false_pos, false_neg, true_neg values
def get_confusion_matrix(t2_sta, threshold, free_indices, fault_indices):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    true_neg = 0
    for pair in fault_indices:
        i, j = pair
        for index in range(i, j + 1):
            if t2_sta[index] > threshold:
                true_pos = true_pos + 1
            else:
                false_neg = false_neg + 1

    for pair in free_indices:
        i, j = pair
        for index in range(i, j + 1):
            if t2_sta[index] > threshold:
                false_pos = false_pos + 1
            else:
                true_neg = true_neg + 1

    return true_pos, false_pos, false_neg, true_neg

def get_confusion_matrix_collective(t2_sta, threshold, free_indices, fault_indices, labels_array, window_size):
    true_pos = false_pos = false_neg = true_neg = 0
    i = 0
    j = window_size
    fault_windows = 0
    free_windows = 0
    while(j <= 11875):
        window = t2_sta[i:j]
        labels_array_window = labels_array[i:j]
        true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = categorize_window(window, labels_array_window, threshold, true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows)
        i += 1
        j += 1
    return true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows

def categorize_window(window, labels_array_window, threshold, true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows):
    window_size = window.size
    fault_points = 0
    free_points = 0
    hamming_distance = 0
    for i in range(0, window_size):
        if labels_array_window[i] == 1:
            fault_points += 1
        else:
            free_points += 1
        if (window[i] >= threshold and labels_array_window[i] == 0) or (window[i] < threshold and labels_array_window[i] == 1):
            hamming_distance += 1
    if fault_points > free_points:
        fault_windows += 1
        if hamming_distance/window_size < 0.2:
            true_pos += 1
        else:
            false_pos += 1
    else:
        free_windows += 1
        if hamming_distance/window_size < 0.2:
            true_neg += 1
        else:
            false_neg += 1
    return true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows

# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance(t2_sta, threshold):
    free_indices, fault_indices = make_labels_list_12000_600_last_batchlen_125()
    true_pos, false_pos, false_neg, true_neg = get_confusion_matrix(t2_sta, threshold, free_indices, fault_indices)
    print(
        'true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score


# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance_50collective(t2_sta, threshold):
    free_indices, fault_indices, labels_array = make_labels_list_12000_600_batchlen125_50collective()
    true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = get_confusion_matrix_50collective(t2_sta, threshold, free_indices, fault_indices, labels_array)
    print(
        '50 collective true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    print(
        '50 collective fault_windows, free_windows are: ' + str(fault_windows) + ', ' + str(free_windows))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score

# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance_100collective(t2_sta, threshold):
    free_indices, fault_indices, labels_array = make_labels_list_12000_600_batchlen125_100collective()
    window_size = 100
    true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = get_confusion_matrix_collective(t2_sta, threshold, free_indices, fault_indices, labels_array, window_size)
    print(
        str(window_size) + ' collective true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    print(
        str(window_size) + ' collective fault_windows, free_windows are: ' + str(fault_windows) + ', ' + str(free_windows))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score

# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance_125collective(t2_sta, threshold):
    free_indices, fault_indices, labels_array = make_labels_list_12000_600_batchlen125_125collective()
    window_size = 125
    true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = get_confusion_matrix_collective(t2_sta, threshold, free_indices, fault_indices, labels_array, window_size)
    print(
        str(window_size) + ' collective true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    print(
        str(window_size) + ' collective fault_windows, free_windows are: ' + str(fault_windows) + ', ' + str(free_windows))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score

# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance_150collective(t2_sta, threshold):
    free_indices, fault_indices, labels_array = make_labels_list_12000_600_batchlen125_150collective()
    window_size = 150
    true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = get_confusion_matrix_collective(t2_sta, threshold, free_indices, fault_indices, labels_array, window_size)
    print(
        str(window_size) + ' collective true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    print(
        str(window_size) + ' collective fault_windows, free_windows are: ' + str(fault_windows) + ', ' + str(free_windows))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score

# labels - boolean array. Corresponding t2 entry is of an anomaly if particular index of 'labels' is true.
def eval_anomaly_detection_performance_200collective(t2_sta, threshold):
    free_indices, fault_indices, labels_array = make_labels_list_12000_600_batchlen125_200collective()
    window_size = 200
    true_pos, false_pos, false_neg, true_neg, fault_windows, free_windows = get_confusion_matrix_collective(t2_sta, threshold, free_indices, fault_indices, labels_array, window_size)
    print(
        str(window_size) + ' collective true_pos, false_pos, false_neg, true_neg respectively are: ' + str(true_pos) + ', ' + str(false_pos) + ', ' + str(false_neg) + ', ' + str(true_neg))
    print(
        str(window_size) + ' collective fault_windows, free_windows are: ' + str(fault_windows) + ', ' + str(free_windows))
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    sensitivity = true_pos / (true_pos + false_neg)
    f1_score = 2 * (recall * precision) / (precision + recall)
    return recall, precision, accuracy, sensitivity, f1_score

def saveRes(name, statistics, threshold, fault_index):
    if not os.path.exists('./result/fig/'):
        os.makedirs('./result/fig/')
    savemat('./result/' + name + '.mat',
            {'statistics': statistics,
             'threshold': threshold,
             'fault_index': fault_index,
             })
    plt.savefig('./result/fig/' + name + '.png', dpi=300)


def fdd_compute(data, threshold, fault_index):
    L = len(data)
    fd_index = L
    for i in range(L - 1, fault_index, -1):
        if (data[i] > threshold):
            fd_index = i - fault_index
    return fd_index


def threshold_compute(statistics):
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import MinMaxScaler

    staMin = np.min(statistics.ravel())
    staMax = np.max(statistics.ravel())
    staBand = (staMax - staMin) * 0.5
    ts = np.linspace(staMin - staBand, staMax + staBand, 1000).reshape(-1, 1)
    kde = np.exp(KernelDensity().fit(statistics.reshape(-1, 1)).score_samples(ts))
    distribution = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.cumsum(kde / 1000).reshape(-1, 1))

    return ts[1000 - len(distribution[distribution > 0.95])]


def plot_ts(test_corr_up, S_up, hidden_view1_up, hidden_view2_up, test_corr_dw, S_dw,
            hidden_view1_dw, hidden_view2_dw, outdim_size, pt_num, num_sample, saveName='None'):
    print(f'DCCA for free data: {-test_corr_up}, DCCA for fault data: {-test_corr_dw}')

    # up_1
    S2_up = np.diag(S_up)
    res1_up = []
    T2_up = []
    Q2_up = []
    Inv_output_up = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_up, S2_up.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_up = hidden_view1_up[i] - np.dot(S2_up, hidden_view2_up[i])
        res1_up.append(te1_up)
        # compute the T2
        te2_up = np.dot(np.dot(te1_up.T, Inv_output_up), te1_up)
        T2_up.append(te2_up)
        # compute the Q2
        q2_up = np.dot(te1_up.T, te1_up)
        Q2_up.append(q2_up)
    res1_up = np.array(res1_up)
    T2_up = np.array(T2_up)
    Q2_up = np.array(Q2_up)
    threshold1_T = get_thred(T2_up)
    # threshold1_Q = get_thred(Q2_up)

    # up_2
    S2_up = np.diag(S_up)
    res2_up = []
    T2_2_up = []
    Q2_2_up = []
    Inv_output_up = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_up, S2_up.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_up = hidden_view2_up[i] - np.dot(S2_up, hidden_view1_up[i])
        res2_up.append(te1_up)
        # compute the T2
        te2_up = np.dot(np.dot(te1_up.T, Inv_output_up), te1_up)
        T2_2_up.append(te2_up)
        # compute the Q2
        q2_up = np.dot(te1_up.T, te1_up)
        Q2_2_up.append(q2_up)
    res2_up = np.array(res2_up)
    T2_2_up = np.array(T2_2_up)
    Q2_2_up = np.array(Q2_2_up)
    threshold_up_2_T = get_thred(T2_2_up)
    # threshold2_Q = get_thred(Q2_2_up)

    # plt.plot(T2_2_up)
    # plt.show()

    # dw_1
    S2_dw = np.diag(S_dw)
    res1_dw = []
    T2_dw = []
    Q2_dw = []
    Inv_output_dw = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_dw, S2_dw.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_dw = hidden_view1_dw[i] - np.dot(S2_dw, hidden_view2_dw[i])
        res1_dw.append(te1_dw)
        # compute the T2
        te2_dw = np.dot(np.dot(te1_dw.T, Inv_output_dw), te1_dw)
        T2_dw.append(te2_dw)
        # compute the Q2
        q2_dw = np.dot(te1_dw.T, te1_dw)
        Q2_dw.append(q2_dw)
    res1_dw = np.array(res1_dw)
    T2_dw = np.array(T2_dw)
    Q2_dw = np.array(Q2_dw)

    # dw_2
    S2_dw = np.diag(S_dw)
    res2_dw = []
    T2_2_dw = []
    Q2_2_dw = []
    Inv_output_dw = np.linalg.inv((np.identity(outdim_size) - np.dot(S2_dw, S2_dw.T)) / (num_sample - 1))
    for i in range(pt_num):
        # compute the residual
        te1_dw = hidden_view2_dw[i] - np.dot(S2_dw, hidden_view1_dw[i])
        res2_dw.append(te1_dw)
        # compute the T2
        te2_dw = np.dot(np.dot(te1_dw.T, Inv_output_dw), te1_dw)
        T2_2_dw.append(te2_dw)
        # compute the Q2
        q2_dw = np.dot(te1_dw.T, te1_dw)
        Q2_2_dw.append(q2_dw)
    res2_dw = np.array(res2_dw)
    T2_2_dw = np.array(T2_2_dw)
    Q2_2_dw = np.array(Q2_2_dw)

    # fault_index = num_sample - 1000 + 1 # check this, hard-coding here
    # fault_index = 10932
    fault_index = 10902
    #fault_index = 11001
    threshold2_T = get_thred(T2_2_dw[0:fault_index])
    #     print('fault index =', fault_index) z
    #     print('threshold =', threshold2_T)

    rate_false_T1, rate_mdr_T1 = far_mdr_compute(T2_dw, threshold1_T, fault_index)
    rate_false_T2, rate_mdr_T2 = far_mdr_compute(T2_2_dw, threshold2_T, fault_index)
    print('Threshold: ' + str(threshold2_T))
    #threshold2_T = threshold2_T / 1.25
    #print('Threshold * 1.25 : ' + str(threshold2_T))

    plt.plot(T2_2_dw)
    plt.show()
    print('Threshold_dw_2: ' + str(threshold2_T))

    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance(T2_2_dw, threshold2_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))

    print('Threshold_up_1: ' + str(threshold1_T))
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance(T2_2_dw, threshold1_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))
    print('Threshold_up_2: ' + str(threshold_up_2_T))
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance(T2_2_dw, threshold_up_2_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))

    thr_reduced = threshold_up_2_T * 0.8
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance(T2_2_dw, thr_reduced)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(precision) + ', ' + str(
        accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))


    """
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance_50collective(T2_2_dw, threshold2_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(
        precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))
    """

    """
    print('Threshold_up_1: ' + str(threshold1_T))
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance_50collective(T2_2_dw, threshold1_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(
        precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))
    """
    """
    # for collective this block of code is enough
    print('Threshold_up_2: ' + str(threshold_up_2_T))
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance_200collective(T2_2_dw, threshold_up_2_T)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(
        precision) + ', ' + str(accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))
    """
    """
    thr_reduced = threshold_up_2_T * 0.8
    recall, precision, accuracy, sensitivity, f1_score = eval_anomaly_detection_performance_50collective(T2_2_dw, thr_reduced)
    print('Recall, precision, accuracy, sensitivity and f1 score respectively: ' + str(recall) + ', ' + str(
        precision) + ', ' + str(
        accuracy) + ', ' + str(sensitivity) + ',' + str(f1_score))
    """

    t = np.linspace(0, pt_num, pt_num)
    fdd = fdd_compute(T2_2_dw, threshold2_T, fault_index)
    smin = np.min(T2_2_dw)
    smax = np.max(T2_2_dw)
    length = len(T2_2_dw)

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 40
    plt.figure(figsize=(16, 8))
    plt.yscale('log')
    plt.plot(np.arange(length), T2_2_dw[:], linewidth=4, alpha=0.7, zorder=0)
    plt.vlines(x=fault_index, ymin=smin, ymax=smax, colors='r', label='fault injection', linestyles='-.', linewidth=8,
               zorder=2)
    plt.hlines(xmin=0, xmax=length, y=threshold2_T, colors='orange', label='detection threshold', linestyles='--',
               linewidth=8, zorder=3)
    print(f'T2_2, thred:{threshold2_T:.3f}, FAR:{rate_false_T2:.2%}, MDR:{rate_mdr_T2:.2%}, FDD:{fdd}')
    plt.ylim([smin, smax])
    plt.legend(loc=4)
    plt.xlabel('Sample')
    plt.ylabel('Test statistic')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    if not (saveName == 'None'):
        saveRes(saveName, T2_2_dw, threshold2_T, fault_index)

    plt.show()


def get_thred(T2_statistic, alpha=0.95):
    # T2_statistic (1, None)
    data = T2_statistic.reshape(-1, 1)
    Min = np.min(data)
    Max = np.max(data)
    Range = Max - Min
    x_start = Min - Range
    x_end = Max + Range
    nums = 2 ** 12
    dx = (x_end - x_start) / (nums - 1)
    data_plot = np.linspace(x_start, x_end, nums)[:, np.newaxis]

    # choose the best bandwidth
    data_median = np.median(data)
    new_median = np.median(np.abs(data - data_median)) / 0.6745
    bw = new_median * ((4 / (3 * data.shape[0])) ** 0.2)

    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)
    log_dens = kde.score_samples(data_plot)
    pdf = np.exp(log_dens).reshape(-1, 1)

    CDF = 0
    index = 0
    while CDF <= alpha:
        CDF += pdf[index] * dx
        index += 1
    return np.squeeze(data_plot[index])
