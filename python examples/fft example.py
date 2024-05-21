# -*- coding: utf-8 -*-
"""
FFT IN PYTHON EXAMPLE
"""

"""
In Python, there are very mature FFT functions both in numpy and scipy. 
In this section, we will take a look of both packages and see how we can easily 
use them in our work. Let’s first generate the signal as before.
"""
import matplotlib.pyplot as plt

import numpy as np

plt.style.use('seaborn-poster')


# sampling rate
sr = 2000
# sampling interval
ts = 1.0/sr
t = np.arange(0,1,ts)

freq = 1.
x = 3*np.sin(2*np.pi*freq*t)

freq = 4
x += np.sin(2*np.pi*freq*t)

freq = 7   
x += 0.5* np.sin(2*np.pi*freq*t)

plt.figure(figsize = (8, 6))
plt.plot(t, x, 'r')
plt.ylabel('Amplitude')

plt.show()

"""
FFT in Numpy¶

EXAMPLE: Use fft and ifft function from numpy to calculate the FFT amplitude 
spectrum and inverse FFT to obtain the original signal. Plot both results. 
Time the fft function using this 2000 length signal.
"""

from numpy.fft import fft, ifft

X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)

plt.stem(freq, np.abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.xlim(0, 10)

plt.subplot(122)
plt.plot(t, ifft(X), 'r')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
"""
Electricity demand in California¶

First, we will explore the electricity demand from California from 2019-11-30 to 2019-12-30.
You can download data from U.S. Energy Information Administration. 
Here, I have already downloaded the data, therefore, we will use it directly.

The electricity demand data from California is stored in ‘930-data-export.csv’ 
in 3 columns. Remember we learned how to read CSV file using numpy. 
Here, we will use another package - pandas, which is a very popular package to 
deal with time series data. We will not teach you this package here, as an 
exercise, you should learn how to use it by yourself. Let us read in the data first.
"""
import pandas as pd

"""
The read_csv function will read in the CSV file. Pay attention to the parse_dates 
parameter, which will find the date and time in column one. 
The data will be read into a pandas DataFrame, we use df to store it. 
Then we will change the header in the original file to something easier to use.
"""

df = pd.read_csv('EIA930_BALANCE_2020_Jul_Dec.csv', 
                 delimiter=',', parse_dates=[1])
df.rename(columns={'UTC Time at End of Hour':'hour',
                   'Demand (MW)':'demand'},
          inplace=True)

plt.figure(figsize = (12, 6))
plt.plot(df['hour'], df['demand'])
plt.xlabel('Datetime')
plt.ylabel('California electricity demand (MWh)')
plt.xticks(rotation=25) 
plt.show()