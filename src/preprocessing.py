import numpy as np
import pandas as pd
import itertools
import pywt

df_train = pd.read_csv("input/train.csv")
df_test  = pd.read_csv("input/test.csv")

def remove_drift():


    hist_bins = np.linspace(-4,10,500)
    clean_hist = []
    hist_bins = np.linspace(-4,10,500)

    train_segm_separators = np.concatenate([[0,500000,600000], np.arange(1000000,5000000+1,500000)])
    train_segm_signal_groups = [0,0,0,1,2,4,3,1,2,3,4]
    train_segm_is_shifted = [False, True, False, False, False, False, False, True, True, True, True]
    train_signal = np.split(df_train['signal'].values, train_segm_separators[1:-1])
    train_opench = np.split(df_train['open_channels'].values, train_segm_separators[1:-1])

    for j,i in enumerate([0,3,4,6,5]):
        clean_hist.append(np.histogram(train_signal[i], bins=hist_bins)[0])
        clean_hist[-1] = clean_hist[-1] / 500000 # normalize histogram


    test_segm_separators = np.concatenate([np.arange(0,1000000+1,100000), [1500000,2000000]])
    test_segm_signal_groups = [0,2,3,0,1,4,3,4,0,2,0,0]
    test_segm_is_shifted = [True, True, False, False, True, False, True, True, True, False, True, False]
    test_signal = np.split(df_test['signal'].values, test_segm_separators[1:-1])


    window_size = 1000
    bin_width = np.diff(hist_bins)[0]
    s_window = 10 # maximum absolute change in shift from window to window+1
    train_signal_shift = []

    for clean_id in range(len(train_segm_signal_groups)):

        group_id = train_segm_signal_groups[clean_id]
        window_shift = []
        prev_s = 0 # all signal groups start with shift=0
        window_data = train_signal[clean_id].reshape(-1,window_size)

        for w in window_data:
            window_hist = np.histogram(w, bins=hist_bins)[0] / window_size
            window_corr = np.array([ np.sum(clean_hist[group_id] * np.roll(window_hist, -s)) for s in range(prev_s-s_window, prev_s+s_window+1) ])
            prev_s = prev_s + np.argmax(window_corr) - s_window
            window_shift.append(-prev_s * bin_width)

        window_shift = np.array(window_shift)
        train_signal_shift.append(window_shift)

        train_signal_shift_clean = []
    train_signal_detrend = []

    for data, use_fit, signal in zip(train_signal_shift, train_segm_is_shifted, train_signal):
        if use_fit:
            data_x = np.arange(len(data), dtype=float) * window_size + window_size/2
            fit = np.flip(np.polyfit(data_x, data, 4))
            data_x = np.arange(len(data) * window_size, dtype=float)
            data_2 = np.sum([ c * data_x ** i for i, c in enumerate(fit) ], axis=0)
        else:
            data_2 = np.zeros(len(data) * window_size, dtype=float)

        train_signal_shift_clean.append(data_2)
        train_signal_detrend.append(signal + data_2)

    #same for test data

    test_signal_detrend = []
    test_signal_shift = []

    for clean_id in range(len(test_segm_signal_groups)):

        group_id = test_segm_signal_groups[clean_id]
        window_shift = []
        prev_s = 0
        window_data = test_signal[clean_id].reshape(-1,window_size)

        for w in window_data:
            window_hist = np.histogram(w, bins=hist_bins)[0] / window_size
            window_corr = np.array([ np.sum(clean_hist[group_id] * np.roll(window_hist, -s)) for s in range(prev_s-s_window, prev_s+s_window+1) ])
            prev_s = prev_s + np.argmax(window_corr) - s_window
            window_shift.append(-prev_s * bin_width)

        window_shift = np.array(window_shift)
        test_signal_shift.append(window_shift)

    test_signal_shift_clean = []
    test_signal_detrend = []
    test_remove_shift = [True, True, False, False, True, False, True, True, True, False, True, False]

    for data, use_fit, signal in zip(test_signal_shift, test_segm_is_shifted, test_signal):
        if use_fit:
            data_x = np.arange(len(data), dtype=float) * window_size + window_size/2
            fit = np.flip(np.polyfit(data_x, data, 4))
            data_x = np.arange(len(data) * window_size, dtype=float)
            data_2 = np.sum([ c * data_x ** i for i, c in enumerate(fit) ], axis=0)
        else:
            data_2 = np.zeros(len(data) * window_size, dtype=float)

        test_signal_shift_clean.append(data_2)
        test_signal_detrend.append(signal + data_2)


    df_train['ndrift_signal'] = list(itertools.chain(*train_signal_detrend))
    df_test['ndrift_signal'] = list(itertools.chain(*test_signal_detrend))

    # df_train.groups = train_segm_signal_groups
    # df_test.groups = test_segm_signal_groups



def denoise():

    def maddest(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def denoise_signal(x, wavelet='db4', level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1/0.6745) * maddest(coeff[-level])

        uthresh = sigma * np.sqrt(2*np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

        return pywt.waverec(coeff, wavelet, mode='per')

    df_train['denoised_signal'] = denoise_signal(df_train.signal)
    df_test['denoised_signal'] = denoise_signal(df_test.signal)

if __name__ == '__main__':
    globals()

    remove_drift()
    denoise()

    df_train.to_csv('input/train.csv',index=False)
    df_test.to_csv('input/test.csv',index=False)
