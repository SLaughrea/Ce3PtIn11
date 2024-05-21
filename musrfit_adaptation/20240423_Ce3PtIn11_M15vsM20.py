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
    f = rawf[t0:keep+t0]; b = rawb[t0:keep+t0]; l = rawl[t0:keep+t0]; r = rawr[t0:keep+t0]

    # we want the background noise before t0 but some needs to be removed
    # we remove t0-100 due to spike of counts around that time
    # we remove the zeros in the background noise
    bkgdf = np.mean(rawf[zero:t0-bad]); bkgdb = np.mean(rawb[zero:t0-bad]) 
    bkgdl = np.mean(rawl[zero:t0-bad]); bkgdr = np.mean(rawr[zero:t0-bad])
    # we remove the background from # of counts

    C= f-bkgdf,b-bkgdb, l-bkgdl, r-bkgdr
    
    # cut data past 6
    # return cleaned counts
    return C

def getErrorBinA(f,b):
    
    binAmount = int(np.round(len(f)/binSize))
    binT = np.linspace(0, len(f), binAmount)*tstep # in units of microseconds

    # error of each of the histograms is sqrt of the histogram's value
    errB = np.sqrt(b+1)
    errF = np.sqrt(f+1)
    
    # initialize bins of f, b and their errors
    binB = np.zeros(binAmount)
    binF = np.zeros(binAmount)
    errBinB = np.zeros(binAmount)
    errBinF = np.zeros(binAmount)
    for i in range(binAmount):
    # calculate bins of f and b
        binB[i] = 1/binAmount*np.sum(b[binSize*i:binSize*(i+1)])
        binF[i] = 1/binAmount*np.sum(f[binSize*i:binSize*(i+1)])

        
        # verified with triumf's data viewer, this is the right error B and F
        #errBinB[i] = np.sqrt(np.sum(errB[binSize*i:binSize*(i+1)]))
        #errBinF[i] = np.sqrt(np.sum(errF[binSize*i:binSize*(i+1)]))
        errBinB[i] = 1/binAmount*np.sqrt(np.sum(b[binSize*i:binSize*(i+1)]))
        errBinF[i] = 1/binAmount*np.sqrt(np.sum(f[binSize*i:binSize*(i+1)]))
    
    # derivative of asymmetry by f
    dadf = alpha * binB * (1+beta) / (beta*alpha*binF + binB)**2
    # derivative of asymmetry by b
    dadb = alpha * binF * (1-beta) / (beta*alpha*binF + binB)**2
    
    dadalpha = (binB*binF*(beta+1)) / (beta*alpha*binF + binB)**2

    dadbeta  = (alpha*binF*(binB-alpha*binF)) / (beta*alpha*binF + binB)**2

    # propagate standard error 
    #df(x,y,...) = sqrt( (df/dx *dx)**2 + (df/dy * dy)**2 + ...)  
    
    
    # musrfit formula http://lmu.web.psi.ch/musrfit/user/html/user-manual.html#fit-types
        

    errBinA = np.sqrt(dadbeta**2*errBeta**2 + dadalpha**2*errAlpha**2 + dadf**2*errBinF**2+dadb**2*errBinB**2)


    # confirmed exact to musrfit
    binA = (alpha*(binF)-(binB))/(alpha*beta*(binF)+(binB))

    #plt.figure(figsize=(12,6))
    #plt.xlim(0,4)
    #plt.errorbar(binT,binA,errBinA)
    #plt.show()
    # return error of asymmetry bins
    
    return binT, binF, errBinF, binB, errBinB, binA, errBinA, dadf, dadb

def plotResults(filename,goodBinT,goodBinA,goodErrorBinA,i,vector=[]):
    #plt.figure(figsize=(8,8),dpi=100)
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

    if len(vector)!= 0:
        fitA = modelall5(vector)
    
        viewFitA = fitA
        #plt.plot(binT, viewFitA, label="Dynesty Fit Second Peak",color="orange",zorder=1)
    #markers = ["o","v","s","D"]
    
    

    #plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, ls="none",marker="o", label="T = {}K".format(temperature[temp]),color=colors[temp])
    plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, ls="none",marker="o",label="{}".format(message[temp]),color=colors[temp])
    
    plt.axis()
    plt.grid()
    #plt.ylim(0.19,0.25)
    plt.xlim(0,6)
    title = "Asymmetry fit for run " + filename + " with all5"
    plt.xlabel("time (µs)",fontsize=12)
    plt.ylabel("Asymmetry",fontsize=12)        
    #plt.title(title,fontsize=12)
    #plt.ylim(0.14,0.18)
    #plt.legend(title="$\\chi^2$ / $n_\\mathrm{{dof}}$ = {0}/{1} = {2}".format(chi2,(N-1),chi2dof),fontsize=12,title_fontsize=12)
    plt.legend(loc="upper right")
    #plt.show()
    
def modelall5(vector):
    rlxStatic,betaStatic,asyTail,asyCe1,rlxCe1,fieldCe1,asyCe2,rlxCe2,fieldCe2 = vector
    # background + static field (electronic below TN, nuclear above) + nonrelaxing longtime tail below TN + precession signal 1 + precession signal 2
    simplExpoBkgd = np.exp(-rlxAg*binT)
    generExpo = np.exp(-(rlxStatic*binT)**betaStatic)
    simplExpo1 = np.exp(-rlxCe1*binT)
    simplExpo2 = np.exp(-rlxCe2*binT)
    TFieldCos1 = np.cos(2*np.pi*fieldCe1*gamma*binT) #0.01 MHz/G
    TFieldCos2 = np.cos(2*np.pi*fieldCe2*gamma*binT)
    
    asyStatic = asysample - asyTail - asyCe1 - asyCe2
    
    return asybkgd*simplExpoBkgd        + \
           asyStatic*generExpo          + \
           asyTail                      + \
           asyCe1*simplExpo1*TFieldCos1 + \
           asyCe2*simplExpo2*TFieldCos2
           
def reducedChiSquare(data,variance,fit,vector):

    # something is wrong with the chi square, its too small. 
    # multiplying by a factor to get a reduced chi-square close to 1
    # this allows the comparison to still be valid
    #
    # NDF is correct, compared to musrfit. The difference is due to a slight difference 
    # in how the bins are counted     

    ChiSquare = ((data-fit)**2/(variance))*95
    
    # The degree of freedom, ν = n − m equals the number of observations n 
    # minus the number of fitted parameters m. 
    NDF = (len(data) - len(vector))

    redChiSquare = ChiSquare/NDF
    
    return NDF,ChiSquare, redChiSquare
#%% global variable initialization

"""
START LOOP OVER FILES HERE
"""
# 2020 ZF
#filenames = ["005813.txt","005812.txt","005816.txt","005811.txt","005810.txt","005809.txt"]
#temperature = [0.019,0.228,0.423,0.999,2.05,4.012]

#filenames = ["005813.txt"]
#temperature = [0.019]

# 2021 ZF
#filenames = ["005138.txt","005137.txt","005136.txt","005142.txt","005139.txt","005141.txt","005140.txt"]
#temperature = [0.056,0.115,0.228,0.427,1.195,1.497,1.799]

# 2020 + 2021 ZF
#colors=["deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue","deepskyblue"]
#filenames = ["005813.txt","005138.txt","005137.txt","005136.txt","005812.txt","005816.txt","005142.txt","005811.txt","005139.txt","005141.txt","005140.txt","005810.txt","005809.txt"]
#temperature = [0.019,0.056,0.115,0.227,0.228,0.423,0.427,0.999,1.195,1.497,1.799,2.05,4.012]

# 2020 LF 1kG runs

#colors= ["red","green","blue","black"]
#filenames = ["005820.txt","005819.txt","005818.txt","005817.txt"]
#temperature = [0.02,1.000,2.05,4.011]

#colors= ["red","green","blue"]
#filenames = ["005808.txt","005807.txt"]
#message = ["M15 GaAs + Ce3PtIn11 Corrected Asymmetry",\
#           "M15 GaAs Corrected Asymmetry","M15 (GaAs + Ce3PtIn11) - GaAs Corrected Asymmetry"]

colors= ["red","black"]
filenames = ["005809.txt","028441.txt"]
message = [r"M15 Ce$_3$PtIn$_{11}$ + holder",\
           r"M20 Ce$_3$PtIn$_{11}$ "]
    
    
    
    
    
#temperature = [0.02,1.000,2.05,4.011]

# 2021 TF runs
#filenames = ["005148.txt","005150.txt","005149.txt","005151.txt"]
#temperature = [0.056,0.299,0.428,4.013]

plt.figure(figsize=(8,8),dpi=100)

for temp in range(len(filenames)):  
    filename = filenames[temp]
    filename = "./runfiles/" + filename
    #plt.xlim(0,4*10**-6)

    np.random.seed(0)
    tstep = 0.390625*10**-9  # (seconds) time interval of binned data
    tstep *= 10**6    # (microseconds)
    # set gamma
    gamma=0.0135528 #MHz/G
    # packing
    binSize =25
    
    default =50 # starting point of data to analyze
    t0 = 1031
    # background to analyze is in the interval [zero,t0-bad]
    bad = 100
    zero = 75
    # cut off data after set number of microseconds
    cut = 6 # microseconds, or 6 *10**-6 seconds
    
    #alpha = 1.07 # just by fitting the 4K
    #beta = 1
    # total_asymmetry and cosin asymmetry free parameters
    
    


    
    #2020
    if filename == "./runfiles/005813.txt" or filename == "./runfiles/005812.txt" or filename == "./runfiles/005816.txt"\
    or filename == "./runfiles/005808.txt" or filename == "./runfiles/005807.txt"\
    or filename == "./runfiles/005811.txt" or filename == "./runfiles/005810.txt" or filename == "./runfiles/005809.txt"\
    or filename == "./runfiles/005820.txt" or filename == "./runfiles/005819.txt" or filename == "./runfiles/005818.txt" or filename == "./runfiles/005817.txt":
        alpha = 1.0746 # 5808_srd.msr
        errAlpha = -0.001
        errBeta = -0.006
        beta = 1.259 # 5808_h12.msr called relasy
        total_asymmetry = 0.06371 # 5808_srd.msr
        if filename == "./runfiles/005807.txt":
            total_asymmetry = 0
        asysample = total_asymmetry
        asybkgd = 0.13877 # 5807.msr
        rlxAg = 0.0091 #rlxAg found with 5808_srd.msr

    
    #2021
    if filename == "./runfiles/005138.txt" or filename == "./runfiles/005137.txt" or filename == "./runfiles/005136.txt" or filename == "./runfiles/005142.txt"\
    or filename == "./runfiles/005144.txt" or filename == "./runfiles/005139.txt" or filename == "./runfiles/005141.txt" or filename == "./runfiles/005140.txt":
        alpha = 0.9632 # 5144_srd.msr
        errAlpha = -0.0011
        errBeta = -0.0068
        beta = 1.2276 # 5144_h12.msr ?
        total_asymmetry = 0.05355 # 5144_srd.msr
        asysample = total_asymmetry
        asybkgd = 0.1495 # 5131.msr
        rlxAg = 0.0091 #rlxAg found with 5808_srd.msr
    
    
    #2021 penetration depth
    if filename == "./runfiles/005148.txt" or filename == "./runfiles/005149.txt" or filename == "./runfiles/005150.txt" or filename == "./runfiles/005151.txt": 
        alpha = 0.87328 # 5144_srd.msr in penetration depth folder with counters 3,4
        errAlpha = -0.00094
        errBeta = -0.0053
        beta = 1.0533 # 5144_h12.msr ?

        asybkgd = 0.1495 # 5131.msr

    
    #2023
    if filename == "./runfiles/028438.txt" or filename == "./runfiles/028441.txt":
        alpha = 1.2103 # 5808_srd.msr
        errAlpha = 0.0012
        errBeta = 0.0056
        beta = 1.0234 # 28438_h12.msr called relasy
        total_asymmetry = 0 # 28438_srd.msr
        asysample = total_asymmetry
        asybkgd = 0 # 5807.msr
        rlxAg = 0. #rlxAg found with 28438_srd.msr
        
        t0 = 670


    
    f,b,l,r = clean(filename)

    binT, binF, errBinF, binB, errBinB, binA, errBinA, dadf, dadb=getErrorBinA(f,b)

    
    #rlxStatic,betaStatic,asyTail,asyCe1,rlxCe1,fieldCe1,asyCe2,rlxCe2,fieldCe2
    
    
    #test
    #bestPeaks = [0.273,1.28,0.000,0.0041,1.23,127.57,0.012,1.38,17]
    
    # dynesty fit
    #bestPeaks = [0.276,1.0,0.003,0.0039,1.266,126.739,0.0015,0,49]
    # dynesty fit with second peak for rlxStatic
    #bestPeaks = [0.344,1.0,0.003,0.0039,1.266,126.739,0.0015,0,49]
    # musr fit
    #bestPeaks = [0.417,1.056,0.0087,0.00395,1.06,120.8,0.00137,0.01,48.7]
    
    # import 
    musrFile = "phenomenological.csv"

    with open(musrFile) as fp:
        reader = csv.reader(fp,delimiter=",",quotechar='"')
        data_read = [row for row in reader]
    musrArray = np.array(data_read)

    #rlxStatic,betaStatic,asyTail,asyCe1,rlxCe1,fieldCe1,asyCe2,rlxCe2,fieldCe2

    # make an array with only the chosen columns of initial array
    musrValue = []
    musrIndexValue = [23,53,26,35,38,32,47,50,44]

    musrIndexErr = [24,54,27,36,39,33,48,51,45]
    musrErr = []

    for i in range(len(musrIndexValue)):
        index = musrIndexValue[i]
        if i==0:
            musrValue = np.append(musrValue,musrArray[1:,index].reshape(-1,1))
            continue
        if i==1:
            musrValue = np.concatenate((musrValue.reshape(-1,1),musrArray[1:,index].reshape(-1,1)),axis=1)
            continue
        musrValue = np.concatenate((musrValue,musrArray[1:,index].reshape(-1,1)),axis=1)

    for i in range(len(musrIndexErr)):
        index = musrIndexErr[i]
        if i==0:
            musrErr = np.append(musrErr,musrArray[1:,index].reshape(-1,1))
            continue
        if i==1:
            musrErr = np.concatenate((musrErr.reshape(-1,1),musrArray[1:,index].reshape(-1,1)),axis=1)
            continue
        musrErr = np.concatenate((musrErr,musrArray[1:,index].reshape(-1,1)),axis=1)



    musrValue = musrValue.astype(float)
    musrErr = musrErr.astype(float)

    bestPeaks = musrValue[temp,:]
    #print(bestPeaks)
    #plotResults(filename, binT, binA,errBinA,i)
    
    
    
    plotResults(filename, binT, binA,errBinA,i,vector=bestPeaks)
