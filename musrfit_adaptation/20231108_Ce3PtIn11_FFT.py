"""
FAST FOURIER TRANSFORM
"""

#%% introduction and import necessary packages
"""
Bayesian Analysis of 005810.txt by Sébastien Laughrea

Some sections will be made into packages for better visibility.
The custom runplot and traceplots are taken from the Dynesty package but had
modifications needed.

The code can be ran as-is to show the comparison between 
internBsl and internFld models.
"""

import numpy as np                     # many useful functions in python
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special        # for bessel or other complex functions
import scipy.signal as signal
import csv

# run Fourier transform of the asymmetry
import scipy
#%% define musr functions
def clean(filename):
    # import data from selected file and remove leading zeros, peak at t0 and 
    # take into account background noise
    
    # length of data worth 6 microseconds, rounded
    keep = int(cut/tstep)
    
    # raw data importation :
    # number of positrons measured on sensors back forward right left
    rawf, rawb, rawr, rawl, todel = np.genfromtxt(
        filename, delimiter=",", skip_header=3, unpack=True)
    # start of the experiment       (bin number)
    # we don't consider data before t0 or after 6 
    f = rawf[t0:keep]; b = rawb[t0:keep]; l = rawl[t0:keep]; r = rawr[t0:keep]
    # we want the background noise before t0 but some needs to be removed
    # we remove t0-100 due to spike of counts around that time
    # we remove the zeros in the background noise
    bkgdf = np.mean(rawf[zero:t0-bad]); bkgdb = np.mean(rawb[zero:t0-bad]) 
    bkgdl = np.mean(rawl[zero:t0-bad]); bkgdr = np.mean(rawr[zero:t0-bad])
    # we remove the background from # of counts

    C= f-bkgdf, b-bkgdb, l-bkgdl, r-bkgdr
    
    # cut data past 6
    # return cleaned counts
    return C

def getErrorBinA(f,b):
    
    binAmount = int(np.round(len(f)/binSize))
    binT = np.linspace(0, len(f), binAmount)*tstep # in units of microseconds

    # error of each of the histograms is sqrt of the histogram's value
    errB = np.sqrt(b)
    errF = np.sqrt(f)
    
    # initialize bins of f, b and their errors
    binB = np.zeros(binAmount)
    binF = np.zeros(binAmount)
    errBinB = np.zeros(binAmount)
    errBinF = np.zeros(binAmount)
    for i in range(binAmount):
    # calculate bins of f and b
        binB[i] = np.mean(b[binSize*i:binSize*(i+1)])
        binF[i] = np.mean(f[binSize*i:binSize*(i+1)])

        # verified with triumf's data viewer, this is the right error B and F
        errBinB[i] = np.sqrt(np.sum(errB[binSize*i:binSize*(i+1)]))
        errBinF[i] = np.sqrt(np.sum(errF[binSize*i:binSize*(i+1)]))
    
    # calculate partial derivative of bin asymmetry by 
    # bin f and bin b
    
    # derivative of asymmetry by f
    dadf = alpha * binB * (1+beta) / (beta*alpha*binF + binB)**2
    # derivative of asymmetry by b
    dadb = alpha * binF * (1-beta) / (beta*alpha*binF + binB)**2

    # propagate standard error 
    #df(x,y,...) = sqrt( (df/dx *dx)**2 + (df/dy * dy)**2 + ...)  
    
    
    # musrfit formula http://lmu.web.psi.ch/musrfit/user/html/user-manual.html#fit-types
    errBinA = np.sqrt((dadf*errBinF)**2+(dadb*errBinB)**2)
    
    
    

    binA = (alpha*(binF)-(binB))/(alpha*beta*(binF)+(binB))
    
    #binA = (binF-binB)/(binF+binB)
    
    #plt.figure(figsize=(12,6))
    #plt.xlim(0,4)
    #plt.errorbar(binT,binA,errBinA)
    #plt.show()
    # return error of asymmetry bins
    return binT, binF, errBinF, binB, errBinB, binA, errBinA

def plotResults(filename,goodBinT,goodBinA,goodErrorBinA,i):
    # draw initial fit with given parameters
    #plt.figure(figsize=(10,6))
    
    # bin data to show for better visibility. Default points to show is a global variable
    # if binSize=50, show 50/100 of bins
    
    if binSize < default:
        viewBinAmount = int(np.round(len(goodBinT)*binSize/default))
        viewGoodBinT = np.zeros(viewBinAmount)
        viewGoodBinA = np.zeros(viewBinAmount)
        viewGoodErrorBinA = np.zeros(viewBinAmount)
        interval = int(np.round(default/binSize))
        for j in range(viewBinAmount):
            viewGoodBinT[j]= np.mean(goodBinT[interval*j:interval*j+interval])
            viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
            if isinstance(goodErrorBinA,float):
                viewGoodErrorBinA= goodErrorBinA
            else:
                viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])

    else:
        viewGoodBinT=goodBinT
        viewGoodBinA=goodBinA
        viewGoodErrorBinA=goodErrorBinA
    """
    y_vals = viewGoodBinA
    y_errs = viewGoodErrorBinA
    N = len(y_vals)
    mu = np.mean(y_vals)
    z = (y_vals - mu) / y_errs
    chi2 = np.sum(z ** 2)
    chi2dof = chi2 / (N - 1)    
    """
    # draw data and fitted line
    #viewGoodErrorBinA = 0
    markers = ["o","v","s","D"]
    plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, ls="none", label="T = {}K".format(temperature[i]), marker = markers[i])#,color="deepskyblue")
    title = "Asymmetry fit for run " + filename + " with all5"
    plt.xlabel("time (µs)",fontsize=12)
    plt.ylabel("Asymmetry",fontsize=12)        
    #plt.title(title,fontsize=12)
    #plt.ylim(0.14,0.18)
    #plt.legend(title="$\\chi^2$ / $n_\\mathrm{{dof}}$ = {0}/{1} = {2}".format(chi2,(N-1),chi2dof),fontsize=12,title_fontsize=12)
    plt.legend()
    #plt.show()
    

#%% global variable initialization

"""
START LOOP OVER FILES HERE
"""
# 2020 ZF
#filenames = ["005813.txt","005812.txt","005816.txt","005811.txt","005810.txt","005809.txt"]
#temperature = [0.019,0.228,0.423,0.999,2.05,4.012]

filenames = ["005813.txt"]
temperature = [0.019]

# 2021 ZF
#filenames = ["005138.txt","005137.txt","005136.txt","005142.txt","005139.txt","005141.txt","005140.txt"]
#temperature = [0.056,0.115,0.228,0.427,1.195,1.497,1.799]

# 2020 + 2021 ZF
#filenames = ["005813.txt","005138.txt","005137.txt","005136.txt","005812.txt","005816.txt","005142.txt","005811.txt","005139.txt","005141.txt","005140.txt","005810.txt","005809.txt"]
#temperature = [0.019,0.056,0.115,0.228,0.228,0.423,0.427,0.999,1.195,1.497,1.799,2.05,4.012]

# 2020 LF 1kG runs
#filenames = ["005820.txt"]#["005820.txt","005819.txt","005818.txt","005817.txt"]
#temperature = [0.02,1.000,2.05,4.011]

# 2021 TF runs
#filenames = ["005148.txt","005150.txt","005149.txt","005151.txt"]
#temperature = [0.056,0.299,0.428,4.013]

plt.figure(figsize=(12,6))
#plt.axis()
plt.grid()
plt.ylim(0.15,0.20)
plt.xlim(0,3)
#plt.xlim(0,4*10**-6)
i=-1
for filename in filenames:  
    i+=1
    np.random.seed(0)
    tstep = 0.390625*10**-9  # (seconds) time interval of binned data
    tstep *= 10**6    # (microseconds)
    # set gamma
    gamma=0.0135528*10**6
    # packing
    binSize =100
    default =100 # starting point of data to analyze
    t0 = 1031
    # background to analyze is in the interval [zero,t0-bad]
    bad = 100
    zero = 75
    # cut off data after set number of microseconds
    cut = 6 # microseconds, or 6 *10**-6 seconds
    
    #alpha = 1.07 # just by fitting the 4K
    #beta = 1
    # total_asymmetry and cosin asymmetry free parameters
    
    alpha = 1.0746
    beta = 1.259
    total_asymmetry = 0.06371
    asysample = total_asymmetry
    asybkgd = 0.13877
    
    
    f,b,l,r = clean(filename)
    binT, binF, errBinF, binB, errBinB, binA, errBinA=getErrorBinA(f,b)
    
    
    # change binT to seconds or keep in microseconds
    binT = binT#/1000000
    """
    from scipy.signal import lombscargle
    dxmin = np.diff(binT).min()
    duration = binT.ptp()
    n = len(binT)
    freqs = np.linspace(1/duration,n/duration,5*n)
    periodogram = lombscargle(binT,binA,freqs)
    kmax = periodogram.argmax()
    """
    
    plotResults(filename, binT, binA,errBinA,i)
    
    """
    t = np.arange(len(f))*tstep # seconds
    
    # get amplitudes of each frequency
    y = np.fft.fft(f)
    m=abs(y)
    
    # frequency in Hz
    fmax = 1/(t[1]-t[0])
    fstep = fmax/len(t)
    freq = np.arange(0,fmax,fstep)
    
    # plot frequencies in Hz, divide by 1E-6 to plot frequencies in MHz
    fftMagExp = plt.figure()
    plt.plot(freq/1000000,m/(len(t)/2))
    plt.xlabel("MHz")
    plt.show()
    
    # zoom in on center of peak
    plt.axis((0.2,5,0,750))
    plt.plot(freq/1000000,m/(len(t)/2))
    plt.xlabel("MHz")
    plt.show()
    """
    """
    # negate the first frequency which is just the mean of the signal
    binA = binA - np.mean(binA)
    # get amplitudes of each frequency
    y = np.fft.fft(binA)
    m=abs(y)
    
    # frequency in Hz
    fmax = 1/(binT[1]-binT[0]) # 1/tstep
    fstep = fmax/len(binT)     # 1/(tstep*len(binT))
    freq = np.arange(0,fmax,fstep)
    
    # plot frequencies in Hz, divide by 1E-6 to plot frequencies in MHz
    fftMagExp = plt.figure()
    plt.plot(freq/1000000,m/(len(binT)/2))
    plt.xlabel("MHz")
    plt.show()
    
    # zoom in on center of peak
    plt.axis((0,6,0,0.003))
    plt.plot(freq/1000000,m/(len(binT)/2),marker=".")
    plt.xlabel("MHz")
    plt.show()
    """
plt.show()





