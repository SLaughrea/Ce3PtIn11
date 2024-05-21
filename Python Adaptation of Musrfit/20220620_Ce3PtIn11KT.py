#%% import necessary packages and assign global variables
import numpy as np                     # many useful functions in python
import matplotlib as mpl
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special
from pylab import cm

import matplotlib.font_manager as fm
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
import matplotlib.font_manager as fm
# Rebuild the matplotlib font cache

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# Generate 2 colors from the 'tab10' colormap
colors = cm.get_cmap('tab10', 2)

# toggle to see all graphs or only fit
showAll = True
titles  = True
# toggle to see only the last fit for the convergence of the alpha
# with the data from the GaAs + Sample run (005808.txt).
# showAll has priority over this
showEachIterationGaAsSample = False
# similar parameter for the GaAs only run (005807.txt)
showEachIterationGaAsOnly   = False
# for GaAs with and without sample
showFinalAsymmetryFit       = True
# for no field runs
showNoField                 = True
# viewed packing
default                     = 200


tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep *= 10**6    # (microseconds)
# set gamma
gamma=0.0135528*10**6
# packing
binSize = 1
# starting point of data to analyze
t0 = 1031
# background to analyze is in the interval [zero,t0-bad]
bad = 100
zero = 75

runParam = [
"0  GaAs      2.282(0.004)K	23 G	                 7118 	 DR",
"1  Ce3PtIn11 4.011(0.001)K	23 G - for zero field	 7596	 DR",
"2  Ce3PtIn11 4.012(0.001)K	zero field	             14315 	 DR",
"3  Ce3PtIn11 2.050(0.001)K	zero field	             10153 	 DR",
"4  Ce3PtIn11 0.999(0.000)K	zero field	             59527 	 DR",
"5  Ce3PtIn11 0.228(0.000)K	zero field	             16275 	 DR",
"6  Ce3PtIn11 0.019(0.000)K	zero field	             14432 	 DR",
"7  Ce3PtIn11 0.019(0.000)K	zero field	             529 	 DR",
"8  Ce3PtIn11 0.019(0.000)K	23 Gauss	             7107 	 DR",
"9  Ce3PtIn11 0.423(0.000)K	0 G	                     12580 	 DR",
"10 Ce3PtIn11 4.011(0.001)K	1 kG	                 12731 	 DR",
"11 Ce3PtIn11 2.050(0.000)K	1 kG	                 10618 	 DR",
"12 Ce3PtIn11 1.000(0.000)K	1 kG	                 11399 	 DR",
"13 Ce3PtIn11 0.020(0.000)K	1 kG	                 12073 	 DR",
"14 Ce3PtIn11 1.000(0.000)K	23 Gauss	             8272 	 DR",
"15 Ce3PtIn11 0.712(0.000)K	0 G	                     10917 	 DR"]

# cut off data after set number of microseconds
cut = 6 # microseconds, or 6 *10**-6 seconds
# iteration number for fitting asymmetry and alpha in GaAs with sample
iteration = 1
# first estimation made from running the 1st iteration with alpha = 1
alpha = 1.1

#%% basic functions and calibration
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

def plotRawAsymmetry(f,asymmetry):
    # time is the x-axis of graphs
    t = np.arange(len(f))*tstep
    if showAll == True:
        # we plot raw asymmetry
        figASS, axASS = plt.subplots()
        axASS.scatter(t,asymmetry,s=1,label="front/back asymmetry",color="deepskyblue")
        axASS.set(xlabel="time (μs)",ylabel="Asymmetry")
        axASS.set_title("Evolution of asymmetry during experiment")
        axASS.legend(loc="lower center",prop={'size': 10},markerscale=5)
        plt.show()

def getErrorBinA(f,b,binRange,binT,binA,filename,alpha):
    asymmetryLabel = filename
    
    # determine number of bins needed 
    binAmount = int(np.round(len(f)/binSize))
    binRange = np.linspace(0, len(f), binAmount)*tstep
    # initialize bins of f and b 
    binB = np.zeros_like(binRange)
    binF = np.zeros_like(binRange)
    # initialize error bins for f and b
    errB = np.zeros_like(binB)
    errF = np.zeros_like(binB)

    
    for j in range(binAmount):
    # calculate bins of f and b
        binB[j] = np.mean(b[binSize*j:binSize*j+binSize])
        binF[j] = np.mean(f[binSize*j:binSize*j+binSize])
        
        # calculate error bins of f and b
        inSQRTB,inSQRTF = 0,0
        #meanB,meanF = np.mean(b[binSize*j:binSize*j+binSize]),np.mean(f[binSize*j:binSize*j+binSize])
        for i in range(binSize):
            if j*binSize+i == len(f):
                break
            
            inSQRTB+=(1/binSize * (np.sqrt(1+b[i+binSize*j])))**2
            inSQRTF+=(1/binSize * (np.sqrt(1+f[i+binSize*j])))**2

        errB[j] = np.sqrt(inSQRTB)
        errF[j] = np.sqrt(inSQRTF)
        
    # calculate partial derivative of asymmetry by f and by b
    dadf = 2*alpha*binB/(binF*alpha+binB)**2
    dadb =-2*alpha*binF/(binF*alpha+binB)**2
    
    # propagate standard error 
    #df(x,y,...) = sqrt( (df/dx *dx)**2 + (df/dy * dy)**2 + ...)   
    errA = np.sqrt((dadf*errF)**2+(dadb*errB)**2)
    
    # return error of asymmetry bins
    return errA

# model 1 is for GaAs with sample
def modelGaAsSample(t,asymmetry1,H,phi,sigma,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-sigma*10**6*t))
    
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsSample(goodBinT,goodBinA,goodErrorBinA,filename):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelGaAsSample)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.22,H=23,phi=0.5,sigma=0.27,p=0.05)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinT) - m.nfit} = {np.round(m.fval/(len(goodBinT) - m.nfit),2)}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    fittedAsymmetry = m.values[0]
    sigma = m.values[3]
    fittedP = m.values[4]
    
    if showAll == True or showEachIterationGaAsSample == True or iteration == 3 and showFinalAsymmetryFit == True:
        # draw initial fit with given parameters
        fig = plt.figure(figsize=(3,3))
        ax = fig.add_axes([0,0,1,1])
        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
        # bin data to show for better visibility. Default points to show is with bin of 100
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
        # draw data and fitted line
        plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(goodBinT, modelGaAsSample(goodBinT, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : sample + GaAs with binning of " + str(binSize) 
        if titles == True:
            plt.title(title,fontsize=12)  
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
    return iteration+1,fittedAsymmetry,sigma,fittedP

# model 2 is for GaAs without sample
def modelGaAsOnly(t,asymmetry1,H,phi,sigma,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-1/2*(sigma*10**6*t)**2))
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsOnly(goodBinT,goodBinA,goodErrorBinA,filename):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelGaAsOnly)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.16,H=23,phi=0,sigma=0.4,p=0.05)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinT) - m.nfit} = {np.round(m.fval/(len(goodBinT) - m.nfit),3)}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    fittedAsymmetry = m.values[0]
    fittedP = m.values[4]

    if showAll == True or showEachIterationGaAsOnly == True or iteration == 3 and showFinalAsymmetryFit == True:
        # draw initial fit with given parameters
        plt.figure(figsize=(10,6))
        
        # bin data to show for better visibility. Default points to show is with bin of 100
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
        # draw data and fitted line
        plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(viewGoodBinT, modelGaAsOnly(viewGoodBinT, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : GaAs only with binning of " + str(binSize)
        if titles == True:
            plt.title(title,fontsize=12)    
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
    return iteration+1,fittedAsymmetry,fittedP

def convergenceGaAsSample(alpha):
    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean("005808.txt")
    # clear the bins for this new calculation
    binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
    # we recalculate asymmetry with correction parameter alpha
    asymmetry = (f*alpha-b)/(f*alpha+b)
    # calculate binned asymmetry
    binAmount = int(np.round(len(asymmetry)/binSize))
    binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
    for j in range(binAmount):
        binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])
    binT=binRange
    binA=binAsymmetry[:len(binRange)]
    # also return yerr on bins
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,"005808.txt",alpha)
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsPlusSampleAsymmetry,sigma,fittedP=fitAsymmetryGaAsSample(goodBinT,goodBinA,goodErrorBinA,"005808.txt")
    alpha = alpha - 2 * fittedP
    return iteration, GaAsPlusSampleAsymmetry,alpha,sigma,binAmount,errorBinA

def convergenceGaAsOnly(alpha):
    # set correct temperature for this run

    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean("005807.txt")

    # clear the bins for this new calculation
    binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
    
    # we recalculate asymmetry with correction parameter alpha
    asymmetry = (f*alpha-b)/(f*alpha+b)

    # calculate binned asymmetry
    binAmount = int(np.round(len(asymmetry)/binSize))
    binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
    for j in range(binAmount):
        binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])

    binT=binRange
    binA=binAsymmetry[:len(binRange)]
        
    # also return yerr on bins
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,"005807.txt",alpha)
    #errorBinA = np.var(binA)           
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsOnlyAsymmetry,fittedP=fitAsymmetryGaAsOnly(goodBinT,goodBinA,goodErrorBinA,"005807.txt")
    
    alpha = alpha - 2 * fittedP
    return iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA

#%% functions for fitting models

def modelCos(t,vector):
        asy1,asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector

        #asy1 = asy_total - asy2 - asy3 - asy4
        simpleGss=np.exp(-1/2*(sigma*binT)**2)
        cosFunction1 = np.cos(2*np.pi*gamma*field1*binT*10**-6+np.pi*phi1/180)
        cosFunction2 = np.cos(2*np.pi*gamma*field2*binT*10**-6+np.pi*phi2/180)
        cosFunction3 = np.cos(2*np.pi*gamma*field3*binT*10**-6+np.pi*phi3/180)

        internFld_signal1 = (amp1*cosFunction1*np.exp(-lambdaT1*binT)+(1-amp1)*np.exp(-lambdaL*binT))
        internFld_signal2 = (amp2*cosFunction2*np.exp(-lambdaT2*binT)+(1-amp2)*np.exp(-lambdaL*binT))
        internFld_signal3 = (amp3*cosFunction3*np.exp(-lambdaT3*binT)+(1-amp3)*np.exp(-lambdaL*binT))

        KuboToyabe = 1/3 + 2/3 * (1-(delta*binT)**2)*np.exp(-delta**2*binT**2/2)
        return (asy1*internFld_signal1+asy2*internFld_signal2+asy3*internFld_signal3+asy4*KuboToyabe)+asy_bkgd*simpleGss  

def modelBessel(t,vector):
        asy1,asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector

        #asy1 = asy_total - asy2 - asy3 - asy4
        simpleGss=np.exp(-1/2*(sigma*binT)**2)
        besselFunction1 = special.jv(0,2*np.pi*gamma*field1*binT*10**-6+np.pi*phi1/180)
        besselFunction2 = special.jv(0,2*np.pi*gamma*field2*binT*10**-6+np.pi*phi2/180)
        besselFunction3 = special.jv(0,2*np.pi*gamma*field3*binT*10**-6+np.pi*phi3/180)

        internFld_signal1 = (amp1*besselFunction1*np.exp(-lambdaT1*binT)+(1-amp1)*np.exp(-lambdaL*binT))
        internFld_signal2 = (amp2*besselFunction2*np.exp(-lambdaT2*binT)+(1-amp2)*np.exp(-lambdaL*binT))
        internFld_signal3 = (amp3*besselFunction3*np.exp(-lambdaT3*binT)+(1-amp3)*np.exp(-lambdaL*binT))

        KuboToyabe = 1/3 + 2/3 * (1-(delta*binT)**2)*np.exp(-delta**2*binT**2/2)
        return (asy1*internFld_signal1+asy2*internFld_signal2+asy3*internFld_signal3+asy4*KuboToyabe)+asy_bkgd*simpleGss  

# fitting no field runs such as 005812.txt
def fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,withBessel=True):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    labels = ["signal_1_asymmetry","signal_2_asymmetry","signal_3_asymmetry","signal_4_asymmetry","field_1","field_2","field_3","fraction_1","fraction_2","fraction_3","phase_1","phase_2","phase_3","lambdaL","lambdaT_1","lambdaT_2","lambdaT_3","delta"]
    vector = 0.01,0.01,0.02,0.03,140,10,1,0.5,0.5,0.5,0,0,0,0.2,1,1,1,0.4
    if withBessel:
        least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelBessel)
    if not withBessel:
        least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelCos)
    m = Minuit(least_squares,vector)
    #asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector
    varMin = [0,0,0,0,120,0,0,0,0,0,0,0,0,0,0,0,0,0]
    varMax = [0.09,0.09,0.09,0.09,250,250,250,1,1,1,360,360,360,5,5,5,5,5]
    print(len(labels), len(varMin), len(varMax), len(vector))
    
    limits = []
    for i in range(len(varMin)):
        limits.append((varMin[i],varMax[i]))
    m.limits = limits
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinT) - m.nfit} = {np.round(m.fval/(len(goodBinT) - m.nfit),3)}",
    ]
    """
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    """
    if showAll == True or showNoField == True:
        # draw initial fit with given parameters
        plt.figure(figsize=(10,6))
        
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
        # draw data and fitted line
        plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        vector=m.values
        if withBessel:
            plt.plot(goodBinT, modelBessel(viewGoodBinT, vector), label="fit",color="orange")
            title = "Asymmetry fit for run " + filename + " with Bessel function fit"
        if not withBessel:
            plt.plot(goodBinT, modelCos(viewGoodBinT, vector), label="fit",color="orange")
            title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        
        if titles == True:
            plt.title(title,fontsize=12)   
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
        for i in range(len(vector)):
            #print(labels[i]," : ", mean[i],"±",cov[i][i])
            #print(labels[i],"\t\t:\t\t", vector[i])
            print("{: >25} {:}".format(labels[i],vector[i]))
        #return fittedParameter1

#%% run parameters, including field strength and run time
# clean forward, backward, left and right counts from raw data
f,b,l,r=clean("005807.txt")
# clear the bins for this new calculation
binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
#%% convergence of alpha and asymmetry calculation for GaAs with sample (run 005808.txt)
# These are the calculations to find the asymmetry due to the GaAs + sample.
# It allows to determine how much the detectors differ, which is the alpha.

# 1st iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)

optimizedAlpha = alpha
#%% convergence of alpha and asymmetry calculation for GaAs only 
#   (run 005807.txt) and for sample only
iteration = 1

# 1st iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)

sampleAsymmetry = GaAsPlusSampleAsymmetry-GaAsOnlyAsymmetry
print("Converged alpha : " + str(np.round(optimizedAlpha,5)))
print("Asymmetry due to sample: " + str(np.round(GaAsPlusSampleAsymmetry,5)) +" - "+ str(np.round(GaAsOnlyAsymmetry,5)) + " = " + str(np.round((sampleAsymmetry),5)))


total_asymmetry = sampleAsymmetry;asy_total=sampleAsymmetry
asy_bkgd=GaAsOnlyAsymmetry

field3 = 1

#%% fitting of bessel and relaxed cosin for nonmagnetic run 


filename = "005140.txt"

# clean forward, backward, left and right counts from raw data
f,b,l,r=clean(filename)
beta = 1.259 # 5808_ ?
# clear the bins for this new calculation
binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])

a = (f*alpha-b)/(f*alpha+b)
asymmetry = ((1-alpha)+a*(1+alpha*beta))/(1+alpha)+a*(1-alpha*beta)
# calculate binned asymmetry
binAmount = int(np.round(len(f)/binSize))
binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
for j in range(binAmount):
    binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])

binT=binRange
binA=asymmetry#binAsymmetry[:len(binRange)]
    
# also return yerr on bins
errorBinA=getErrorBinA(f,b,binRange,binT,binA,filename,alpha)
#errorBinA = np.var(binA)           
goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA

# following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,withBessel=True)

fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,withBessel=False)



