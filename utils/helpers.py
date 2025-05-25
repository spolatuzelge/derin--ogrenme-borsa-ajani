import numpy as np

"""
Belirtilen indisteki kapanış fiyatı, SMA_5, SMA_20 ve getiriyi döndürür.
"""
def get_state(data, index):
    return np.array([
        float(data.loc[index, 'Close']),
        float(data.loc[index, 'SMA_5']),
        float(data.loc[index, 'SMA_20']),
        float(data.loc[index, 'Returns'])
    ])