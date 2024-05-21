import numpy as np                     # many useful functions in python
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special

# toggle to see all graphs or only fit
showAll = True

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
# choose if lambdaL is kept fixed at a value of 0
fixedLambdaL                = False
# viewed packing
default                     = 200


tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep *= 10**6    # (microseconds)
# set gamma
gamma=0.0135528*10**6
# packing
binSize = 50
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
        plt.title(title)
        plt.show()
    return x,y

def getErrorBinY(f,b,binRange,binX,binY,filename,temperature,alpha):
    binAmount=len(binRange)
    asymmetryLabel = filename + ": T = " + temperature + "K"
    # the error is the variance, or the square of the standard deviation
    #asymmetry = (f*alpha-b)/(f*alpha+b)
    #varA = np.var(asymmetry)
    #errorBinY = varA * 1/np.sqrt(len(f)-1)
    
    t = np.arange(len(f))*tstep
    asymmetry = (f*alpha-b)/(f*alpha+b)
    errorA = 0
    for k in range(len(f)):
        errorA += (asymmetry[k]-np.mean(asymmetry))**2
    varA = 1/len(f)*errorA
    #print(varA,np.var(asymmetry)) # this is identical to np.var
    plt.errorbar(t,asymmetry,varA,ecolor="red",markersize = 1)
    plt.show()
    errorBinY = np.zeros_like(binY)
    errorBinY = varA /binSize**(1/2)
    print("Variance of binned ")
    #%%
    """
    dadf=-(f*alpha-b)/(f*alpha+b)**2+1/(f*alpha+b)
    dadb=-(f*alpha-b)/(f*alpha+b)**2-1/(f*alpha+b)

    
    
    # calculate variance f and b
    meanF = np.mean(f)
    errorF = 0
    for k in range(len(f)):
        errorF+=(f[k]-meanF)**2
    varF=errorF/(len(f)-1)
    

    meanB = np.mean(b)
    errorB = 0
    for k in range(len(b)):
        errorB+=(b[k]-meanB)**2
    varB=errorB/(len(b)-1)
    
    
    errorFB = 0
    for k in range(len(f)):
        errorFB+=(f[k]-meanF)*(b[k]-meanB)
    varFB = errorFB/(len(f)-1)
    
    # propagate uncertainty to calculate variance of asymmetry
    varA =1/len(f)*( varB * np.mean(dadb)**2 + varF * np.mean(dadf)**2 + varFB * np.mean(dadf) * np.mean(dadb) )
    """
    
    """
    for j in range(len(binY)):

        meanVar=np.mean(varA[(j*binSize):(j*binSize+binSize)])

        errorBinY[j]=np.sqrt(1/binSize)*meanVar
    """
    
    # propagate uncertainty for the appropriate binning
    """
    errorBinY = np.zeros_like(binY)
    # vertical error of each bin
    for j in range(len(binY)):
        sumEA2=0
        for k in range(binSize):
            if j*binSize+k == len(f):
                break
            sumEA2+=varA[(j*binSize+k)]
            print(sumEA2)
        errorBinY[j]=np.sqrt(1/binSize)*sumEA2
    """    
    #%%

    if showAll == True:
        plt.xlabel("time (μs)")
        plt.ylabel("Asymmetry")
        plt.title("Bins with error bars")
        plt.plot(binRange, np.zeros(binAmount),color="deepskyblue",linestyle="--")
        plt.errorbar(binX,binY,errorBinY,fmt=".",label=asymmetryLabel,color="deepskyblue",ecolor="red",markersize = 1)
        plt.legend(loc="lower center")
        plt.show()
    return errorBinY

# model 1 is for GaAs with sample
def model1(t,asymmetry1,H,phi,lambda1,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-lambda1*10**6*t))
    
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsSample(goodBinX,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinX values to seconds for fitting
    goodBinX = goodBinX*10**-6
    least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model1)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.22,H=23,phi=0,lambda1=0.4,p=0.05)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinX) - m.nfit} = {np.round(m.fval/(len(goodBinX) - m.nfit),2)}",
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
            
            viewBinAmount = int(np.round(len(goodBinX)*binSize/default))

            viewGoodBinX = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodBinX[j]= np.mean(goodBinX[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodBinX=goodBinX
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodBinX, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(goodBinX, model1(goodBinX, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : sample + GaAs with binning of " + str(binSize) 
        plt.title(title,fontsize=12)  
        plt.legend(title="\n".join(fit_info),fontsize=12,title_fontsize=12);
        plt.show()
    return iteration+1,fittedAsymmetry,fittedP

# model 2 is for GaAs without sample
def model2(t,asymmetry1,H,phi,sigma,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-1/2*(sigma*10**6*t)**2))
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsOnly(goodBinX,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinX values to seconds for fitting
    goodBinX = goodBinX*10**-6
    least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model2)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.07,H=23,phi=0,sigma=0.4,p=0.05)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinX) - m.nfit} = {np.round(m.fval/(len(goodBinX) - m.nfit),2)}",
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
            
            viewBinAmount = int(np.round(len(goodBinX)*binSize/default))

            viewGoodBinX = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodBinX[j]= np.mean(goodBinX[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodBinX=goodBinX
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodBinX, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(viewGoodBinX, model2(viewGoodBinX, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        title = "Asymmetry fit for run " + filename + " : GaAs only with binning of " + str(binSize)
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
    binX=binRange
    binY=binAsymmetry[:len(binRange)]
        
    # also return yerr on bins
    errorBinY=getErrorBinY(f,b,binRange,binX,binY,"005808.txt",temperature,alpha)
            
    goodBinX, goodBinA, goodErrorBinA = binX, binY, errorBinY
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsPlusSampleAsymmetry,fittedP=fitAsymmetryGaAsSample(goodBinX,goodBinA,goodErrorBinA,"005808.txt",temperature)
    alpha = alpha - 2 * fittedP
    return iteration, GaAsPlusSampleAsymmetry,alpha,binAmount

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

    binX=binRange
    binY=binAsymmetry[:len(binRange)]
        
    # also return yerr on bins
    errorBinY=getErrorBinY(f,b,binRange,binX,binY,"005807.txt",temperature,alpha)
            
    goodBinX, goodBinA, goodErrorBinA = binX, binY, errorBinY
    
    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsOnlyAsymmetry,fittedP=fitAsymmetryGaAsOnly(goodBinX,goodBinA,goodErrorBinA,"005807.txt",temperature)
    
    alpha = alpha - 2 * fittedP
    return iteration, GaAsOnlyAsymmetry, alpha, binAmount


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
def fitAsymmetryNoField(goodBinX,goodBinA,goodErrorBinA,filename,temperature,withBessel=True):
    # switch back goodBinX values to seconds for fitting
    goodBinX = goodBinX*10**-6
    
    # starting values
    
    if fixedLambdaL:
        if withBessel:
            least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model3)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,sigma=-60000)
        if not withBessel:
            least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model4)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,sigma=60000)
    if not fixedLambdaL:
        if withBessel :
            least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model5)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,lambdaL=40000,sigma=-0.36)
        if not withBessel:
            least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model6)
            m = Minuit(least_squares,beta=0.05,H=140,phi=0,lambdaT=40000,lambdaL=-40000,sigma=0.306)
        
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinX) - m.nfit} = {np.round(m.fval/(len(goodBinX) - m.nfit),2)}",
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
            viewBinAmount = int(np.round(len(goodBinX)*binSize/default))
            viewGoodBinX = np.zeros(viewBinAmount)
            viewGoodBinA = np.zeros(viewBinAmount)
            viewGoodErrorBinA = np.zeros(viewBinAmount)
            interval = int(np.round(default/binSize))
            for j in range(viewBinAmount):
                viewGoodBinX[j]= np.mean(goodBinX[interval*j:interval*j+interval])
                viewGoodBinA[j]= np.mean(goodBinA[interval*j:interval*j+interval])
                if isinstance(goodErrorBinA,float):
                    viewGoodErrorBinA= goodErrorBinA
                else:
                    viewGoodErrorBinA[j]= np.mean(goodErrorBinA[interval*j:interval*j+interval])
        
        else:
            viewGoodBinX=goodBinX
            viewGoodBinA=goodBinA
            viewGoodErrorBinA=goodErrorBinA
        # draw data and fitted line
        plt.errorbar(viewGoodBinX, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
        if fixedLambdaL:
            if withBessel:
                plt.plot(viewGoodBinX, model3(viewGoodBinX, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with Bessel function fit"
            if not withBessel:
                plt.plot(viewGoodBinX, model4(viewGoodBinX, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
        if not fixedLambdaL:
            if withBessel:
                plt.plot(viewGoodBinX, model5(viewGoodBinX, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with Bessel function fit"
            if not withBessel:
                plt.plot(viewGoodBinX, model6(viewGoodBinX, *m.values), label="fit",color="orange")
                title = "Asymmetry fit for run " + filename + " with relaxed cosin fit"
        plt.xlabel("time (s)",fontsize=12)
        plt.ylabel("Asymmetry",fontsize=12)
        
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
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount = convergenceGaAsSample(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount = convergenceGaAsSample(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry, alpha, binAmount = convergenceGaAsSample(alpha)

#print("Converged alpha for GaAs with sample: " + str(np.round(alpha,5)))
#print("Asymmetry of GaAs with sample for this alpha: " + str(np.round(GaAsPlusSampleAsymmetry,5)))

optimizedAlpha = alpha
#%% convergence of alpha and asymmetry calculation for GaAs only 
#   (run 005807.txt) and for sample only
iteration = 1

# 1st iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount = convergenceGaAsOnly(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount = convergenceGaAsOnly(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsOnlyAsymmetry, alpha, binAmount = convergenceGaAsOnly(alpha)

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

binX=binRange
binY=binAsymmetry[:len(binRange)]
    
# also return yerr on bins
errorBinY=getErrorBinY(f,b,binRange,binX,binY,"005812.txt",temperature,alpha)
        
goodBinX, goodBinA, goodErrorBinA = binX, binY, errorBinY

# following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
fitAsymmetryNoField(goodBinX,goodBinA,goodErrorBinA,"005812.txt",temperature,withBessel=True)
fitAsymmetryNoField(goodBinX,goodBinA,goodErrorBinA,"005812.txt",temperature,withBessel=False)



