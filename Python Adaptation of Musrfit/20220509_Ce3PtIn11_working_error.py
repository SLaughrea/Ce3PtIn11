import numpy as np                     # many useful functions in python
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special

# toggle to see all graphs or only fit
showAll = False
titles  = False
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
# choose if lambdaL is kept fixed at a value of 0
fixedLambdaL                = False
# viewed packing
default                     = 200


tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep *= 10**6    # (microseconds)
# set gamma
gamma=0.0135528*10**6
# packing
binSize = 25
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
# assignment of global variables
# there were 16 runs done at TRIUMF during November 2020 on the 1952 experiment
filename = ["005807.txt", "005808.txt", "005809.txt", "005810.txt", "005811.txt",
             "005812.txt", "005813.txt", "005814.txt", "005815.txt", "005816.txt",
             "005817.txt", "005818.txt", "005819.txt", "005820.txt", "005821.txt",
             "005822.txt"]
# cut off data after set number of microseconds
cut = 6 # microseconds, or 6 *10**-6 seconds
# iteration number for fitting asymmetry and alpha in GaAs with sample
iteration = 1
# first estimation made from running the 1st iteration with alpha = 1
alpha = 1.1

#%% functions
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

def plotCounts(f,b,temperature):
    # time is the x-axis of graphs
    t = np.arange(len(f))*tstep
    if showAll == True:
        # plotting number of forward and backward counts over duration
        figRAW, axRAW = plt.subplots()
        #color=next(axRAW._get_lines.prop_cycler)['color']
        axRAW.scatter(t,f,s=1,label="forward counts for "+temperature+"K")
        axRAW.scatter(t,b,s=1,label="backward counts for "+temperature+"K")
        axRAW.set(xlabel="time (μs)",ylabel="# of counts per bin")
        if titles == True:
            axRAW.set_title("Forward and backward counts")
        axRAW.legend(loc="upper right",prop={'size': 10},markerscale=5)
        plt.show()

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

def plotBinnedAsymmetry(f,binRange,binAsymmetry,filename,temperature):
    asymmetryLabel = filename + ": T = " + temperature + "K"
    x=binRange
    y=binAsymmetry[:len(binRange)]
    binAmount=len(binRange)
    if showAll == True:
        plt.scatter(x,y,s=2, label=asymmetryLabel,color="deepskyblue")
        plt.plot(binRange, np.zeros(binAmount),color="deepskyblue",linestyle="--")
        plt.xlabel("time (μs)")
        plt.ylabel("Asymmetry")
        title = "Evolution of binned asymmetry during experiment"
        plt.legend(loc="lower center")
        if titles == True:
            plt.title(title)
        plt.show()
    return x,y

def getErrorBinA(f,b,binRange,binT,binA,filename,temperature,alpha):
    asymmetryLabel = filename + ": T = " + temperature + "K"
    
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

    """
    plt.figure(figsize=(10,6))
    plt.errorbar(binT,binB,errB,fmt=".",ecolor="red")
    plt.show()
    plt.figure(figsize=(10,6))
    plt.errorbar(binT,binA,errA,fmt=".",ecolor="red")
    plt.show()
    """

    # plot bins with error bars if asked
    if showAll == True:
        plt.xlabel("time (μs)")
        plt.ylabel("Asymmetry")
        if titles == True:
            plt.title("Bins with error bars")
        plt.plot(binRange, np.zeros(binAmount),color="deepskyblue",linestyle="--")
        plt.errorbar(binT,binA,errA,fmt=".",label=asymmetryLabel,color="deepskyblue",ecolor="red",markersize = 1)
        plt.legend(loc="lower center")
        plt.show()
    
    # return error of asymmetry bins
    return errA

# model 1 is for GaAs with sample
def model1(t,asymmetry1,H,phi,sigma,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-sigma*10**6*t))
    
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsSample(goodBinT,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model1)
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
    fittedP = m.values[4]
    #for k in range(len(fit_info)):
    #    print("\n"+fit_info[k])
    
    # iteration number, for showing results at last (3rd) iteration
    
    
    if showAll == True or showEachIterationGaAsSample == True or iteration == 3 and showFinalAsymmetryFit == True:
        # draw initial fit with given parameters
        plt.figure(figsize=(10,6))
        
        # bin data to show for better visibility. Default points to show is with bin of 100
        # if binSize=50, show 50/100 of bins

        if binSize < default:
            
            viewBinAmount = int(np.round(len(goodBinT)*binSize/default))

            viewGoodbinT = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodbinT[j]= np.mean(goodBinT[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodbinT=goodBinT
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodbinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(goodBinT, model1(goodBinT, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : sample + GaAs with binning of " + str(binSize) 
        if titles == True:
            plt.title(title,fontsize=12)  
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
    return iteration+1,fittedAsymmetry,fittedP

# model 2 is for GaAs without sample
def model2(t,asymmetry1,H,phi,sigma,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-1/2*(sigma*10**6*t)**2))
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsOnly(goodBinT,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model2)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.16,H=23,phi=0,sigma=0.4,p=0.05)    
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
    fittedP = m.values[4]
    #for k in range(len(fit_info)):
    #    print("\n"+fit_info[k])

    if showAll == True or showEachIterationGaAsOnly == True or iteration == 3 and showFinalAsymmetryFit == True:
        # draw initial fit with given parameters
        plt.figure(figsize=(10,6))
        
        # bin data to show for better visibility. Default points to show is with bin of 100
        # if binSize=50, show 50/100 of bins
        if binSize < default:
            
            viewBinAmount = int(np.round(len(goodBinT)*binSize/default))

            viewGoodbinT = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodbinT[j]= np.mean(goodBinT[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodbinT=goodBinT
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodbinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(viewGoodbinT, model2(viewGoodbinT, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : GaAs only with binning of " + str(binSize)
        if titles == True:
            plt.title(title,fontsize=12)    
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
    return iteration+1,fittedAsymmetry,fittedP

def convergenceGaAsSample(alpha):
    # set correct temperature for this run
    temperature = runParam[1][13:18]
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
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,"005808.txt",temperature,alpha)
        
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsPlusSampleAsymmetry,fittedP=fitAsymmetryGaAsSample(goodBinT,goodBinA,goodErrorBinA,"005808.txt",temperature)
    alpha = alpha - 2 * fittedP
    return iteration, GaAsPlusSampleAsymmetry,alpha,binAmount,errorBinA

def convergenceGaAsOnly(alpha):
    # set correct temperature for this run
    temperature = runParam[1][13:18]
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
    errorBinA=getErrorBinA(f,b,binRange,binT,binA,"005807.txt",temperature,alpha)
    #errorBinA = np.var(binA)           
    goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA
    
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsOnlyAsymmetry,fittedP=fitAsymmetryGaAsOnly(goodBinT,goodBinA,goodErrorBinA,"005807.txt",temperature)
    
    alpha = alpha - 2 * fittedP
    return iteration, GaAsOnlyAsymmetry, alpha, binAmount,errorBinA


# model 3 is for 005812 (zero field) with the bessel function with fixed lambdaL = 0
def model3(t,beta,H,phi,lambdaT,sigma):
    lambdaL=0
    simpleGss=np.exp(-1/2*(sigma*t)**2)
    besselFunction = special.jv(0,2*np.pi*gamma*H*t+np.pi*phi/180)
    return sampleAsymmetry*(beta*besselFunction*np.exp(-lambdaT*t)+(1-beta)*np.exp(-lambdaL*t))+GaAsOnlyAsymmetry*simpleGss
    
# model 4 is for 005812 (zero field) with the relaxed cosin function with fixed lambdaL = 0
def model4(t,beta,H,phi,lambdaT,sigma):
    lambdaL=0
    simpleGss=np.exp(-1/2*(sigma*t)**2)
    cosFunction = np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)
    #cosFunction2 = np.cos(2*np.pi*gamma*H*t+np.pi*phi2/180)
    #cosFunction3 = np.cos(2*np.pi*gamma*H*t+np.pi*phi3/180)
    return sampleAsymmetry*(beta*cosFunction*np.exp(-lambdaT*t)+(1-beta)*np.exp(-lambdaL*t))+GaAsOnlyAsymmetry*simpleGss

# model 5 is for 005812 (zero field) with the bessel function with lambdaL free
def model5(t,beta,H,phi,lambdaT,lambdaL,sigma):
    simpleGss=np.exp(-1/2*(sigma*t)**2)
    besselFunction = special.jv(0,2*np.pi*gamma*H*t+np.pi*phi/180)
    return sampleAsymmetry*(beta*besselFunction*np.exp(-lambdaT*t)+(1-beta)*np.exp(-lambdaL*t))+GaAsOnlyAsymmetry*simpleGss
    
# model 6 is for 005812 (zero field) with the relaxed cosin function with lambdaL free
def model6(t,beta,H,phi,lambdaT,lambdaL,sigma):
    simpleGss=np.exp(-1/2*(sigma*t)**2)
    cosFunction = np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)
    #cosFunction2 = np.cos(2*np.pi*gamma*H*t+np.pi*phi2/180)
    #cosFunction3 = np.cos(2*np.pi*gamma*H*t+np.pi*phi3/180)
    return sampleAsymmetry*(beta*cosFunction*np.exp(-lambdaT*t)+(1-beta)*np.exp(-lambdaL*t))+GaAsOnlyAsymmetry*simpleGss


# fitting no field runs such as 005812.txt
def fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,filename,temperature,withBessel=True):
    # switch back goodBinT values to seconds for fitting
    goodBinT = goodBinT*10**-6
    
    # starting values
    
    if fixedLambdaL:
        if withBessel:
            least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model3)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,sigma=-60000)
        if not withBessel:
            least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model4)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,sigma=60000)
    if not fixedLambdaL:
        if withBessel :
            least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model5)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,lambdaL=40000,sigma=-0.36)
        if not withBessel:
            least_squares = LeastSquares(goodBinT,goodBinA,goodErrorBinA,model6)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,lambdaL=-40000,sigma=0.306)
        
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
        
    # how to extract fitted data
    #fittedParameter1 = m.values[0]

    #for k in range(len(fit_info)):
    #    print("\n"+fit_info[k])

    if showAll == True or showNoField == True:
        # draw initial fit with given parameters
        plt.figure(figsize=(10,6))
        
        # bin data to show for better visibility. Default points to show is with bin of 100
        # if binSize=50, show 50/100 of bins

        if binSize < default:
            viewBinAmount = int(np.round(len(goodBinT)*binSize/default))
            viewGoodbinT = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodbinT[j]= np.mean(goodBinT[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodbinT=goodBinT
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodbinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        if fixedLambdaL:
            if withBessel:
                plt.plot(viewGoodbinT, model3(viewGoodbinT, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with Bessel function fit"
            if not withBessel:
                plt.plot(viewGoodbinT, model4(viewGoodbinT, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
        if not fixedLambdaL:
            if withBessel:
                plt.plot(viewGoodbinT, model5(viewGoodbinT, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with Bessel function fit"
            if not withBessel:
                plt.plot(viewGoodbinT, model6(viewGoodbinT, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        
        if titles == True:
            plt.title(title,fontsize=12)   
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
        #return fittedParameter1

#%% run parameters, including field strength, temperature, and run time
# set correct temperature for this run
temperature = runParam[1][13:18]
# clean forward, backward, left and right counts from raw data
f,b,l,r=clean("005807.txt")
# clear the bins for this new calculation
binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
#%% convergence of alpha and asymmetry calculation for GaAs with sample (run 005808.txt)
# These are the calculations to find the asymmetry due to the GaAs + sample.
# It allows to determine how much the detectors differ, which is the alpha.

# 1st iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsSample(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsSample(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount,errorBinA = convergenceGaAsSample(alpha)

#print("Converged alpha for GaAs with sample: " + str(np.round(alpha,5)))
#print("Asymmetry of GaAs with sample for this alpha: " + str(np.round(GaAsPlusSampleAsymmetry,5)))

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

#print("Converged alpha for GaAs only: " + str(np.round(alpha,5)))
#print("Asymmetry of GaAs only for this alpha: " + str(np.round(GaAsOnlyAsymmetry,5)))
sampleAsymmetry = GaAsPlusSampleAsymmetry-GaAsOnlyAsymmetry
print("Converged alpha : " + str(np.round(optimizedAlpha,5)))
print("Asymmetry due to sample: " + str(np.round(GaAsPlusSampleAsymmetry,5)) +" - "+ str(np.round(GaAsOnlyAsymmetry,5)) + " = " + str(np.round((sampleAsymmetry),5)))


#%% fitting of bessel and relaxed cosin for nonmagnetic run 005812.txt


# set correct temperature for this run
temperature = runParam[5][13:18]
# clean forward, backward, left and right counts from raw data
f,b,l,r=clean("005812.txt")

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
errorBinA=getErrorBinA(f,b,binRange,binT,binA,"005812.txt",temperature,alpha)
#errorBinA = np.var(binA)           
goodBinT, goodBinA, goodErrorBinA = binT, binA, errorBinA

# following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,"005812.txt",temperature,withBessel=True)
fitAsymmetryNoField(goodBinT,goodBinA,goodErrorBinA,"005812.txt",temperature,withBessel=False)



