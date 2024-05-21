#%% import necessary packages and assign global variables
import numpy as np                     # many useful functions in python
import matplotlib as mpl
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special
#from pylab import cm

import matplotlib.font_manager as fm
# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
fm = mpl.font_manager
# Rebuild the matplotlib font cache
#fm.get_cachedir()
# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2

# Generate 2 colors 
#colors = ("deepskyblue","orange")
colors = ("black","red")
# toggle to see all graphs or only fit
showAll = False
showTitles  = True
# toggle to see only the last fit for the convergence of the alpha
# with the data from the GaAs + Sample run (005808.txt).
# showAll has priority over this
showEachIterationGaAsSample = False
# similar parameter for the GaAs only run (005807.txt)
showEachIterationGaAsOnly   = False
# for GaAs with and without sample
showFinalAsymmetryFit       = False
# for no field runs
showNoField                 = True
# print specific sets of data into txt format
# default and binSize must be the same or code must be adapted
printDATA                   = False
# viewed packing
default                     = 100


tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep *= 10**6    # (microseconds)
# set gamma
gamma=0.0135538817*10**6 # 0.0135538817 MHz/G
# packing
binSize = 100
# starting point of data to analyze
t0 = 1031
# background to analyze is in the interval [zero,t0-bad]
bad = 100
zero = 75
"""
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
"""
path = "C:/Users/User/OneDrive - Universite de Montreal/Masters/Codes/RunFiles/"
all_runs = ["005807.txt","005808.txt","005809.txt","005810.txt","005811.txt","005812.txt","005813.txt","005814.txt","005815.txt","005816.txt","005817.txt","005818.txt","005819.txt","005820.txt","005821.txt","005822.txt"]
all_runs = [path + s for s in all_runs]
chosen_runs = ["005809.txt","005810.txt","005811.txt","005812.txt","005813.txt","005816.txt","005822.txt"]
chosen_runs = [path + s for s in chosen_runs]
all_temps = [2.282,4.011,4.012,2.050,0.999,0.228,0.019,0.019,0.019,0.423,4.011,2.050,1.000,0.020,1.000,0.712]

dict_runs_temp = dict(zip(all_runs,all_temps))
# cut off data after set number of microseconds
cut = 6 # microseconds, or 6 *10**-6 seconds
# iteration number for fitting asymmetry and alpha in GaAs with sample
iteration = 1
# first estimation made from running the 1st iteration with alpha = 1
alpha = 1.1
sigma = None

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
    f = rawf[t0:keep+t0]; b = rawb[t0:keep+t0]; l = rawl[t0:keep+t0]; r = rawr[t0:keep+t0]
    # we want the background noise before t0 but some needs to be removed
    # we remove t0-100 due to spike of counts around that time
    # we remove the zeros in the background noise
    bkgdf = np.mean(rawf[zero:t0-bad]); bkgdb = np.mean(rawb[zero:t0-bad]) 
    bkgdl = np.mean(rawl[zero:t0-bad]); bkgdr = np.mean(rawr[zero:t0-bad])
    # we remove the background from # of counts
    C= f-bkgdf, b-bkgdb, l-bkgdl, r-bkgdr

    # return cleaned counts
    return C

def getErrorBinA(f,b,binRange,binT,binA,filename,alpha):
    
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
        # draw initial fit with given parameters
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0,0,1,1])
        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        # Edit the major and minor tick locations
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1e-6))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5*1e-6))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
        # Set the axis limits
        ax.set_xlim(0, goodBinT[-1])
        ax.set_ylim(-0.4, 0.4)
        ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color=colors[0])
        ax.plot(goodBinT, modelGaAsSample(goodBinT, *m.values), label="fit",color=colors[1])
        ax.set_xlabel("time (s)",labelpad=10)
        ax.set_ylabel("Asymmetry",labelpad=10)
        ax.ticklabel_format(style="sci", useMathText = True)
        filename = filename[-10:-4]
        title = "Asymmetry fit for run " + filename + " : sample + GaAs with binning of " + str(binSize) 
        if showTitles == True:
            plt.title(title)  
        #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        ax.legend()
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
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0,0,1,1])
        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
        # Edit the major and minor tick locations
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1e-6))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5*1e-6))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
        # Set the axis limits
        ax.set_xlim(0, goodBinT[-1])
        ax.set_ylim(-0.4, 0.4)
        ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color=colors[0])
        ax.plot(goodBinT, modelGaAsOnly(goodBinT, *m.values), label="fit",color=colors[1])
        ax.set_xlabel("time (s)",labelpad=10)
        ax.set_ylabel("Asymmetry",labelpad=10)
        ax.ticklabel_format(style="sci", useMathText = True)
        
        filename = filename[-10:-4]
        title = "Asymmetry fit for run " + filename + " : sample + GaAs with binning of " + str(binSize) 
        if showTitles == True:
            plt.title(title)  
        #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        ax.legend()
        plt.show()
        
    return iteration+1,fittedAsymmetry,fittedP

def convergenceGaAsSample(alpha):
    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean(path+"005808.txt")
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
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,path+"005808.txt",alpha)
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsPlusSampleAsymmetry,sigma,fittedP=fitAsymmetryGaAsSample(goodBinT,goodBinA,goodErrorBinA,"005808.txt")
    alpha = alpha - 2 * fittedP
    return iteration, GaAsPlusSampleAsymmetry,alpha,sigma,binAmount,errorBinA

def convergenceGaAsOnly(alpha):
    # set correct temperature for this run

    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean(path+"005807.txt")

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
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,path+"005807.txt",alpha)
    #errorBinA = np.var(binA)           
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsOnlyAsymmetry,fittedP=fitAsymmetryGaAsOnly(goodBinT,goodBinA,goodErrorBinA,path+"005807.txt")
    
    alpha = alpha - 2 * fittedP
    return iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA

#%% functions for fitting models

def modelCos(t,vector):
        global asy_bkgd
        global sigma

        asy1,asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector

        #asy1 = total_asymmetry - asy2 - asy3 - asy4
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
        global asy_bkgd
        global sigma

        asy1,asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector

        #asy1 = total_asymmetry - asy2 - asy3 - asy4
        simpleGss=np.exp(-1/2*(sigma*binT)**2)
        besselFunction1 = special.jv(0,2*np.pi*gamma*field1*binT*10**-6+np.pi*phi1/180)
        besselFunction2 = special.jv(0,2*np.pi*gamma*field2*binT*10**-6+np.pi*phi2/180)
        besselFunction3 = special.jv(0,2*np.pi*gamma*field3*binT*10**-6+np.pi*phi3/180)

        internFld_signal1 = (amp1*besselFunction1*np.exp(-lambdaT1*binT)+(1-amp1)*np.exp(-lambdaL*binT))
        internFld_signal2 = (amp2*besselFunction2*np.exp(-lambdaT2*binT)+(1-amp2)*np.exp(-lambdaL*binT))
        internFld_signal3 = (amp3*besselFunction3*np.exp(-lambdaT3*binT)+(1-amp3)*np.exp(-lambdaL*binT))

        KuboToyabe = 1/3 + 2/3 * (1-(delta*binT)**2)*np.exp(-delta**2*binT**2/2)
        return (asy1*internFld_signal1+asy2*internFld_signal2+asy3*internFld_signal3+asy4*KuboToyabe)+asy_bkgd*simpleGss  

def modelKT3f9(t, vector):
    

        global asy_bkgd
        global sigma

        asy1,asy2, asy3, field1, field2, amp1, amp2, phi1, phi2, lambdaL, lambdaT1, lambdaT2, delta  = vector

        #asy1 = total_asymmetry - asy2 - asy3 - asy4
        simpleGss=np.exp(-1/2*(sigma*binT)**2)

        cosFunction1 = np.cos(2*np.pi*gamma*field1*binT*10**-6+np.pi*phi1/180)
        cosFunction2 = np.cos(2*np.pi*gamma*field2*binT*10**-6+np.pi*phi2/180)
        internFld_signal1 = (amp1*cosFunction1*np.exp(-lambdaT1*binT)+(1-amp1)*np.exp(-lambdaL*binT))
        internFld_signal2 = (amp2*cosFunction2*np.exp(-lambdaT2*binT)+(1-amp2)*np.exp(-lambdaL*binT))


        KuboToyabe = 1/3 + 2/3 * (1-(delta*binT)**2)*np.exp(-delta**2*binT**2/2)
        #return (asy1*internFld_signal1+asy2*internFld_signal2+asy3*internFld_signal3+asy4*KuboToyabe)+asy_bkgd*simpleGss  
        return (asy1+internFld_signal1+asy2*internFld_signal2+asy3*KuboToyabe+asy_bkgd*simpleGss)

# fitting no field runs such as 005812.txt
def fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,model):
    global binT
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    limits = []
    if model == "Bessel":
        labels = ["Signal 1 Asymmetry","Signal 2 Asymmetry","Signal 3 Asymmetry","Signal 4 Asymmetry","Field 1","Field 2","Field 3","Fraction 1","Fraction 2","Fraction 3","Phase 1","Phase 2","Phase 3","Lambda L","Lambda T1","Lambda T2","Lambda T3","Delta"]
        vector = 0.01,0.01,0.02,0.03,140,10,1,0.5,0.5,0.5,0,0,0,0.2,1,1,1,0.4
        #asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector
        varMin = [0,0,0,0,120,0,0,0,0,0,0,0,0,0,0,0,0,0]
        varMax = [0.09,0.09,0.09,0.09,250,250,250,1,1,1,360,360,360,5,5,5,5,5]
        least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelBessel)
    if model == "cos_KT":
        labels = ["Signal 1 Asymmetry","Signal 2 Asymmetry","Signal 3 Asymmetry","Signal 4 Asymmetry","Field 1","Field 2","Field 3","Fraction 1","Fraction 2","Fraction 3","Phase 1","Phase 2","Phase 3","Lambda L","Lambda T1","Lambda T2","Lambda T3","Delta"]
        vector = 0.01,0.01,0.02,0.03,140,10,1,0.5,0.5,0.5,0,0,0,0.2,1,1,1,0.4
        #asy2, asy3, asy4, field1, field2, field3, amp1, amp2, amp3, phi1, phi2, phi3, lambdaL, lambdaT1, lambdaT2, lambdaT3, delta  = vector
        varMin = [0,0,0,0,120,0,0,0,0,0,0,0,0,0,0,0,0,0]
        varMax = [0.09,0.09,0.09,0.09,250,250,250,1,1,1,360,360,360,5,5,5,5,5]
        least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelCos)
    if model == "KT3f9":
        labels = ["Signal 1 Asymmetry","Signal 2 Asymmetry","Signal 3 Asymmetry","Field 1","Field 2","Fraction 1","Fraction 2","Phase 1","Phase 2","Lambda L","Lambda T1","Lambda T2","Delta"]
        vector = 0.035, 0.026,0.0096,115,51,0.15,0.063,-18,-53,0.394,1.18,0,0.35
        #asy1,asy2, asy3, field1, field2, amp1, amp2, phi1, phi2, lambdaL, lambdaT1, lambdaT2, delta  = vector
        varMin = [0,0,0,0,0,0,0,-180,-180,0,0,0,0.2]
        varMax = [0.0734,0.0734,0.0734,250,250,1,1,180,180,5,5,5,0.8]
        least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,modelKT3f9)
    
    m = Minuit(least_squares,vector)
    for i in range(len(varMin)):
        limits.append((varMin[i],varMax[i]))
    m.limits = limits
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    """
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinT) - m.nfit} = {np.round(m.fval/(len(goodBinT) - m.nfit),3)}",
    ]
    
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    """
    if showAll == True or showNoField == True:
        # draw initial fit with given parameters
        
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
        #plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        vector=m.values
        if model=="Bessel_KT":
            fitbinned = modelBessel(viewGoodBinT,vector)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_axes([0,0,1,1])
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            # Edit the major and minor tick locations
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1e-6))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5*1e-6))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
            # Set the axis limits
            ax.set_xlim(0, goodBinT[-1])
            ax.set_ylim(0.12, 0.32)
            ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="Sample",color=colors[0])
            ax.plot(goodBinT, modelBessel(viewGoodBinT, vector), label="Fit with Bessel zero \nfield function at "+str(dict_runs_temp[filename])+" K",color=colors[1])
            ax.set_xlabel("time (s)",labelpad=10)
            ax.set_ylabel("Asymmetry",labelpad=10)
            ax.ticklabel_format(style="sci", useMathText = True)
            filename = filename[-10:-4]
            title = "Asymmetry fit for run " + filename + " with Bessel function fit" 
            if showTitles == True:
                plt.title(title)  
            #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
            ax.legend()
            plt.savefig(title+".pdf",bbox_inches='tight')
            plt.show()
            
        if model == "cos_KT":
            fitbinned = modelCos(viewGoodBinT,vector)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_axes([0,0,1,1])
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            # Edit the major and minor tick locations
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1e-6))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5*1e-6))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
            # Set the axis limits
            ax.set_xlim(0, goodBinT[-1])
            ax.set_ylim(0.12, 0.32)
            ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="Sample",color=colors[0])
            ax.plot(goodBinT, modelCos(viewGoodBinT, vector), label="Fit with cosin zero \nfield function at "+str(dict_runs_temp[filename])+" K",color=colors[1])
            ax.set_xlabel("time (s)",labelpad=10)
            ax.set_ylabel("Asymmetry",labelpad=10)
            ax.ticklabel_format(style="sci", useMathText = True)
            filename = filename[-10:-4]
            title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
            if showTitles == True:
                plt.title(title)  
            #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
            ax.legend()
            plt.savefig(title+".pdf",bbox_inches='tight')
            plt.show()

        if model == "KT3f9":
            fitbinned = modelKT3f9(viewGoodBinT,vector)
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_axes([0,0,1,1])
            # Edit the major and minor ticks of the x and y axes
            ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
            ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
            ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
            ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
            # Edit the major and minor tick locations
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1e-6))
            ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5*1e-6))
            ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
            # Set the axis limits
            ax.set_xlim(0, goodBinT[-1])
            ax.set_ylim(0.12, 0.32)
            ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="Sample",color=colors[0])
            ax.plot(goodBinT, modelKT3f9(viewGoodBinT, vector), label="Fit with KT3f_9 zero \nfield function at "+str(dict_runs_temp[filename])+" K",color=colors[1])
            ax.set_xlabel("time (s)",labelpad=10)
            ax.set_ylabel("Asymmetry",labelpad=10)
            ax.ticklabel_format(style="sci", useMathText = True)
            filename = filename[-10:-4]
            title = "Asymmetry fit for run " + filename + " with KT3f_9 fit"
            if showTitles == True:
                plt.title(title)  
            #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
            ax.legend()
            plt.savefig(title+".pdf",bbox_inches='tight')
            plt.show()
        
        #for i in range(len(vector)):
            #print(labels[i]," : ", mean[i],"±",cov[i][i])
            #print(labels[i],"\t\t:\t\t", vector[i])
            #print("{: >25} {:}".format(labels[i],vector[i]))
        if printDATA == True:
            if (filename == "005809" or filename == "005811" or filename == "005813"):
                """
                with open("time"+filename+".txt", "w") as output:
                    for row in viewGoodBinT:
                        s = "".join(map(str, str(row)))
                        output.write(s+'\n')
                with open("asymmetry"+filename+".txt", "w") as output:
                    for row in viewGoodBinA:
                        s = "".join(map(str, str(row)))
                        output.write(s+'\n')
                with open("asymmetryerrorbar"+filename+".txt", "w") as output:
                    for row in viewGoodErrorBinA:
                        s = "".join(map(str, str(row)))
                        output.write(s+'\n')
                with open("asymmetryfit"+filename+".txt", "w") as output:
                    for row in viewGoodErrorBinA:
                        s = "".join(map(str, str(row)))
                        output.write(s+'\n')
                """
                
                
                with open("data"+filename+".txt","w") as output:
                    output.write("time\tasymmetry\tasymmetry_errorbar\tasymmetry_fit\n")
                    for i in range(len(viewGoodBinT)):
                        s = "".join(map(str, str(viewGoodBinT[i])))
                        output.write(s+'\t')
                        s = "".join(map(str, str(viewGoodBinA[i])))
                        output.write(s+'\t')
                        s = "".join(map(str, str(viewGoodErrorBinA[i])))
                        output.write(s+'\t')
                        s = "".join(map(str, str(fitbinned[i])))
                        output.write(s+'\t')
                        output.write("\n")
                #print("PRINTED")
    return labels,m

# calculate alpha from GaAsOnly and calculate total_asymmetry and asy_bkgd from GaAsPlusSample
def initialization():
    global alpha
    global sigma
    global asy_bkgd
    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean(path+"005807.txt")
    # convergence of alpha and asymmetry calculation for GaAs with sample (run 005808.txt)
    # These are the calculations to find the asymmetry due to the GaAs + sample.
    # It allows to determine how much the detectors differ, which is the alpha.
    # 1st iteration for convergence of alpha
    iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)
    
    # 2nd iteration for convergence of alpha
    iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)
    
    # 3rd iteration for convergence of alpha
    iteration, GaAsPlusSampleAsymmetry, alpha,sigma, binAmount,errorBinA = convergenceGaAsSample(alpha)
    

    # convergence of alpha and asymmetry calculation for GaAs only 
    #   (run 005807.txt) and for sample only
    iteration = 1
    
    # 1st iteration for convergence of alpha
    iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)
    
    # 2nd iteration for convergence of alpha
    iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)
    
    # 3rd iteration for convergence of alpha
    iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsOnly(alpha)
    
    sampleAsymmetry = GaAsPlusSampleAsymmetry-GaAsOnlyAsymmetry
    #print("Converged alpha : " + str(np.round(alpha,5)))
    #print("Asymmetry due to sample: " + str(np.round(GaAsPlusSampleAsymmetry,5)) +" - "+ str(np.round(GaAsOnlyAsymmetry,5)) + " = " + str(np.round((sampleAsymmetry),5)))
    
    
    total_asymmetry = sampleAsymmetry
    asy_bkgd=GaAsOnlyAsymmetry

    field3 = 1
    return total_asymmetry, asy_bkgd, field3

#%% fitting of bessel and relaxed cosin for nonmagnetic run 
def prepare(filename):
    global binT
    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean(filename)
    
    # clear the bins for this new calculation
    binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
    
    asymmetry = (f*alpha-b)/(f*alpha+b)
    
    # calculate binned asymmetry
    binAmount = int(np.round(len(f)/binSize))
    binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
    for j in range(binAmount):
        binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])
    
    binT=binRange
    binA=binAsymmetry[:len(binRange)]
        
    # also return yerr on bins
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,filename,alpha)
    #errorBinA = np.var(binA)           
    
    return binT, binA, errorBinA
        
def fitModel(filename,model):
    goodBinT, goodBinA, goodErrorBinA=prepare(filename)
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html        
    return fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,model)

# create library for optimized values and errors of each parameter for each run file
def getDictionaryRuns(model):
    models = []     # one for each temperature
    for i in range(len(chosen_runs)):
        parameters, val = fitModel(chosen_runs[i],model)
        models.append(val)
    dict_runs = {}
    for i in range(len(models)):
        model_params = {}
        for j in range(models[i].npar):
            #model_params.append({"labels": labels[j],"value": model.values[j], "error":model.errors[j]})
            model_params[parameters[j]]={"value": models[i].values[j], "error":models[i].errors[j]}
        dict_runs[chosen_runs[i]]=model_params
    
    
    # create dictionary of dictionaries for each parameter. The dictionary of a single parameter contains
    # each associated temperature
    dict_parameters = {}

    # fill dictionary with dictionaries of each parameter at different temperatures
    for i in parameters:
        # create dictionary of a single parameter at different temperatures
        dict_parameter = {}
        
        for j in chosen_runs:
            # dictionary[4 K] = {"value": dict_runs[4K]}
            value = dict_runs.get(j).get(i).get("value")
            error = dict_runs.get(j).get(i).get("error")
            dict_parameter[dict_runs_temp.get(j)] = {"value": value, "error": error}
        
        # add the dictionary of single parameter to the main one
        dict_parameters[i] = dict_parameter
        
        # return parameter names, dictionary of each parameter with varying temperatures, dictionary of each temperature with varying variable
    return parameters, dict_parameters, dict_runs

def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()

def plotParameterOverTime(parameter,dict_parameter):
    X = sorted(list(dict_parameter.keys()))
    Y = [dict_parameter.get(key).get("value") for key in X]
    dY = [dict_parameter.get(key).get("error") for key in X]
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_axes([0,0,1,1])
    # Edit the major and minor ticks of the x and y axes
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='off')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='off')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='off')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='off')
    # Edit the major and minor tick locations
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
    #ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.01))
    # Set the axis limits
    ax.set_xlim(0, 5.000001)
    #ax.set_ylim(0.12, 0.32)
    #ax.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="Sample",color=colors[0])
    ax.errorbar(X, Y, dY, fmt=".",color=colors[0],markersize=10)
    ax.set_xlabel("Temperature (K)",labelpad=10)
    ax.set_ylabel(parameter,labelpad=10)
    ax.ticklabel_format(style="sci", useMathText = True)
    title = "Temperature Dependence of " +  parameter
    if showTitles == True:
        plt.title(title)  
    #ax.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
    plt.savefig(title+".pdf",dpi=600,bbox_inches='tight')
    plt.show()
#%% main

# Calculate alpha, total and background asymmetry.
initialization()  

# Get parameter names, dictionary of each parameter with varying temperatures, 
# and dictionary of each temperature with varying variable.

# This function uses the chosen_runs and the chosen model to optimize parameters to data of the chosen runs
# models can be "cos_KT", "KT3f9", "Bessel"
parameters, dict_parameters, dict_runs = getDictionaryRuns(model="KT3f9")

for i in parameters:
    plotParameterOverTime(i,dict_parameters.get(i))











