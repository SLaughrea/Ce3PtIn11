import numpy as np                     # many useful functions in python
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error

# toggle to see all graphs or only fit
showAll = False

# toggle to see only the last fit for the convergence of the alpha
# with the data from the GaAs + Sample run (005808.txt).
# showAll has priority over this
showEachIterationGaAsSample = True


#%% functions
def clean(filename):
    # import data from selected file and remove leading zeros, peak at t0 and 
    # take into account background noise

    # raw data importation :
    # number of positrons measured on sensors back forward right left
    rawf, rawb, rawr, rawl, todel = np.genfromtxt(
        filename, delimiter=",", skip_header=3, unpack=True)
    # start of the experiment       (bin number)
    # we don't consider data before t0
    f = rawf[t0:]; b = rawb[t0:]; l = rawl[t0:]; r = rawr[t0:]
    # we want the background noise before t0 but some needs to be removed
    # we remove t0-100 due to spike of counts around that time
    # we remove the zeros in the background noise
    bkgdf = np.mean(rawf[zero:t0-bad]); bkgdb = np.mean(rawb[zero:t0-bad]) 
    bkgdl = np.mean(rawl[zero:t0-bad]); bkgdr = np.mean(rawr[zero:t0-bad])
    # we remove the background from # of counts
    C= f-bkgdf, b-bkgdb, l-bkgdl, r-bkgdr
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

    asymmetryLabel = filename + ": T = " + temperature + "K"
    # error on each asymmetry point
    errorA = np.zeros_like(f)
    dadf=-(f-alpha*b)/(f+alpha*b)**2+1/(f+alpha*b)
    dadb=-(f-alpha*b)/(f+alpha*b)**2-1/(f+alpha*b)
    for j in range(len(f)):
        if dadf[j]**2*f[j]+dadb[j]**2*b[j] < 0:
            errorA[j]=0
        else:
            errorA[j]=np.sqrt(dadf[j]**2*f[j]+dadb[j]**2*b[j])
    
    errorBinY = np.zeros_like(binY)
    # vertical error of each bin
    for j in range(len(binY)):
        sumEA2=0
        for k in range(binSize):
            if j*binSize+k == len(errorA):
                break
            sumEA2+=errorA[j*binSize+k]**2
        errorBinY[j]=1/binSize*np.sqrt(sumEA2)
    if showAll == True:
        plt.xlabel("time (μs)")
        plt.ylabel("Asymmetry")
        plt.title("Bins with error bars")
        plt.plot(binRange, np.zeros(binAmount),color="deepskyblue",linestyle="--")
        plt.errorbar(binX,binY,errorBinY,fmt=".",label=asymmetryLabel,color="deepskyblue")
        plt.legend(loc="lower center")
        plt.show()
        
    return errorBinY

def cutData(cut,f,asymmetry,binX,binY,errorBinY,filename,temperature):    
    # length of data worth 6 microseconds, rounded
    keep = int(cut/tstep)
    # binned data is binSize smaller
    keepBin = int(cut/tstep/binSize)
    # cut asymmetry, binned asymmetry and error
    goodA, goodBinA, goodErrorBinA = asymmetry[:keep], binY[:keepBin], errorBinY[:keepBin]
    goodBinX=binX[:keepBin]
    asymmetryLabel = filename + ": T = " + temperature + "K"
    if showAll == True:
        plt.xlabel("time (μs)")
        plt.ylabel("Asymmetry")
        plt.title("Bins with error bars up to 6 μs")
        plt.plot(goodBinX, np.zeros(len(goodBinX)),color="deepskyblue",linestyle="--")
        plt.errorbar(goodBinX,goodBinA,goodErrorBinA,fmt=".",label=asymmetryLabel,color="deepskyblue")
        plt.legend(loc="lower center")
        plt.show()
    return goodA, goodBinA,goodBinX, goodErrorBinA
  
"""  
def model(t,asymettry1,beta,H,phi,lambdaL,lambdaT,asymettry2):
    gamma=0.0135528*10**6 #Hz/G
    besselFunction = special.jv(0,2*np.pi*gamma*H*t+np.pi*phi/180)
    return asymettry1*(beta*besselFunction*np.exp(-lambdaT*t)+(1-beta)*np.exp(-lambdaL*t))+asymettry2
"""

# model 1 is for GaAs with sample
def model(t,asymmetry1,H,phi,lambda1,p):
    relaxcos=asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-lambda1*t))
    return (p+relaxcos)/(1+p*relaxcos)

def fitAsymmetryGaAsSample(goodBinX,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinX values to seconds for fitting
    goodBinX = goodBinX*10**-6
    least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.22,H=23,phi=0,lambda1=40000,p=0.05)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinX) - m.nfit}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    fittedAsymmetry = m.values[0]
    fittedP = m.values[4]
    #for k in range(len(fit_info)):
    #    print("\n"+fit_info[k])
    
    # iteration number, for showing results at last (3rd) iteration
    
    
    if showAll == True or showEachIterationGaAsSample == True or iteration == 3 :
        # draw initial fit with given parameters
        plt.figure(figsize=(12,8))
        #plt.plot(goodBinX,model(goodBinX,asymmetry1=0.22,H=23,phi=0,lambda1=40000,p=0.05),label="initial",linestyle="--",color="red")
        # draw data and fitted line
        plt.errorbar(goodBinX, goodBinA, goodErrorBinA, fmt=".", label="data",color="deepskyblue")
        plt.plot(goodBinX, model(goodBinX, *m.values), label="fit",color="orange")
        plt.xlabel("time (s)")
        plt.ylabel("Asymmetry")
        if iteration >=1:
            title = "Asymmetry fit for run" + filename + " : iteration #" + str(iteration)
        else:
            title = "Asymmetry fit for run" + filename + " : initial fit for alpha guess"
        plt.title(title)    
        plt.legend(title="\n".join(fit_info));
        plt.show()
    return iteration+1,fittedAsymmetry,fittedP

# model 2 is for GaAs without sample
def model2(t,asymmetry1,H,phi,lambda1):
    gamma=0.0135528*10**6
    return asymmetry1*(np.cos(2*np.pi*gamma*H*t+np.pi*phi/180)*np.exp(-lambda1*t))

def fitAsymmetryGaAsOnly(goodBinX,goodBinA,goodErrorBinA,filename,temperature):
    # switch back goodBinX values to seconds for fitting
    goodBinX = goodBinX*10**-6
    least_squares = LeastSquares(goodBinX,goodBinA,goodErrorBinA,model2)
    # starting values
    m = Minuit(least_squares,asymmetry1=0.07,H=23,phi=0,lambda1=40000)    
    # finds minimum of least_squares function
    m.migrad()
    # accurately computes uncertainties
    m.hesse()
    
    # display legend with some fit info
    fit_info = [
        f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(goodBinX) - m.nfit}",
    ]
    for p, v, e in zip(m.parameters, m.values, m.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    fittedAsymmetry = m.values[0]
    
    #for k in range(len(fit_info)):
    #    print("\n"+fit_info[k])

    # draw initial fit with given parameters
    plt.figure(figsize=(12,8))
    #plt.plot(goodBinX,model(goodBinX,asymmetry1=0.22,H=23,phi=0,lambda1=40000,p=0.05),label="initial",linestyle="--",color="red")
    # draw data and fitted line
    plt.errorbar(goodBinX, goodBinA, goodErrorBinA, fmt=".", label="data",color="deepskyblue")
    plt.plot(goodBinX, model2(goodBinX, *m.values), label="fit",color="orange")
    plt.xlabel("time (s)")
    plt.ylabel("Asymmetry")
    title = "Asymmetry fit for run" + filename
    plt.title(title)    
    plt.legend(title="\n".join(fit_info));
    plt.show()
    return fittedAsymmetry

def convergenceGaAsSample(alpha):


    # clear the bins for this new calculation
    binAsymmetry = np.zeros([300])
    # set correct temperature for this run
    temperature = runParam[1][13:18]
    # clean forward, backward, left and right counts from raw data
    f,b,l,r=clean("005808.txt")

    # we recalculate asymmetry with correction parameter alpha
    asymmetry = (f-alpha*b)/(f+alpha*b)

    # calculate binned asymmetry
    binAmount = int(np.round(len(asymmetry)/binSize))
    binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
    for j in range(binAmount):
        binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])

    # plot positron count
    plotCounts(f,b,temperature)

    # plot raw asymmetry
    plotRawAsymmetry(f,asymmetry)

    # also return asymmetry bins positions
    binX,binY=plotBinnedAsymmetry(f,binRange,binAsymmetry,"005808.txt",temperature)
        
    # also return yerr on bins
    errorBinY=getErrorBinY(f,b,binRange,binX,binY,"005808.txt",temperature,alpha)
            
    # cut data past 6 
    # cut asymmetry, binned asymmetry (y positions (A) and x positions), and yerr
    goodA, goodBinA, goodBinX, goodErrorBinA = cutData(cut,f,asymmetry,binX,binY,errorBinY,"005808.txt",temperature)

    # following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
    iteration,GaAsPlusSampleAsymmetry,fittedP=fitAsymmetryGaAsSample(goodBinX,goodBinA,goodErrorBinA,"005808.txt",temperature)
    alpha = alpha + 2 * fittedP
    return iteration, GaAsPlusSampleAsymmetry,alpha

#%% run parameters, including field strength, temperature, and run time
#0         1         2         3         4         5         6         7      "
#01234567890123456789012345678901234567890123456789012345678901234567890123456"
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
#%% assignment of global variables
# there were 16 runs done at TRIUMF during November 2020 on the 1952 experiment
filename = ["005807.txt", "005808.txt", "005809.txt", "005810.txt", "005811.txt",
             "005812.txt", "005813.txt", "005814.txt", "005815.txt", "005816.txt",
             "005817.txt", "005818.txt", "005819.txt", "005820.txt", "005821.txt",
             "005822.txt"]

# "005807.txt" : GaAs run to get the "background" asymmetry
# "005808.txt" : GaAs + Ce3PtIn11 run to get the background + sample asymmetry
# we can then isolate this asymmetry

tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep = tstep * 10**6    # (microseconds)

# set gamma
gamma=0.0135528*10**6

# assign temperature for each run based on run parameters
temperature = runParam[0][13:18]
# binned asymmetry for each run
binAsymmetry = np.zeros([300])
binSize = 100

# starting point of data to analyze
t0 = 1031
# background to analyze is in the interval [zero,t0-bad]
bad = 100
zero = 75
# cut off data after set number of microseconds
cut = 6 # microseconds, or 6 *10**-6 seconds

# iteration number for fitting asymmetry and alpha in GaAs with sample
iteration = 0
    
# first estimation
alpha = 0.9
#%% convergence of alpha and asymmetry calculation for GaAs with sample (run 005808.txt)
# These are the calculations to find the asymmetry due to the GaAs + sample.
# It allows to determine how much the detectors differ, which is the alpha.

# initial guess of alpha
iteration, GaAsPlusSampleAsymmetry,alpha=convergenceGaAsSample(alpha)

# 1st iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry,alpha=convergenceGaAsSample(alpha)

# 2nd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry,alpha=convergenceGaAsSample(alpha)

# 3rd iteration for convergence of alpha
iteration, GaAsPlusSampleAsymmetry,alpha=convergenceGaAsSample(alpha)

#%% asymmetry calculation for GaAs only (run 005807.txt) and for sample only

print("Converged alpha for GaAs with sample: " + str(np.round(alpha,5)))
print("Asymmetry of GaAs with sample for this alpha: " + str(np.round(GaAsPlusSampleAsymmetry,5)))

# clear the bins for this new calculation
binAsymmetry = np.zeros([300])
# set correct temperature for this run
temperature = runParam[0][13:18]
# clean forward, backward, left and right counts from raw data
f,b,l,r=clean("005807.txt")

# we calculate asymmetry with correction parameter alpha
asymmetry = (f-alpha*b)/(f+alpha*b)

# calculate binned asymmetry
binAmount = int(np.round(len(asymmetry)/binSize))
binRange = np.linspace(0, len(asymmetry), binAmount)*tstep
for j in range(binAmount):
    binAsymmetry[j]= np.mean(asymmetry[binSize*j:binSize*j+binSize])

# plot positron count
plotCounts(f,b,temperature)

# plot raw asymmetry
plotRawAsymmetry(f,asymmetry)

# also return asymmetry bins positions
binX,binY=plotBinnedAsymmetry(f,binRange,binAsymmetry,"005807.txt",temperature)
    
# also return yerr on bins
errorBinY=getErrorBinY(f,b,binRange,binX,binY,"005807.txt",temperature,alpha)
        
# cut data past 6 
cut = 6 # microseconds, or 6 *10**-6 seconds
# cut asymmetry, binned asymmetry (y positions (A) and x positions), and yerr
goodA, goodBinA, goodBinX, goodErrorBinA = cutData(cut,f,asymmetry,binX,binY,errorBinY,"005807.txt",temperature)

# following instructions on https://iminuit.readthedocs.io/en/stable/tutorial/basic_tutorial.html
GaAsOnlyAsymmetry=fitAsymmetryGaAsOnly(goodBinX,goodBinA,goodErrorBinA,"005807.txt",temperature)

print("Asymmetry of GaAs with sample for this alpha: " + str(np.round(GaAsOnlyAsymmetry,5)))
print("Asymmetry due to sample: " + str(np.round(GaAsPlusSampleAsymmetry,5)) +" - "+ str(np.round(GaAsOnlyAsymmetry,5)) + " = " + str(np.round((GaAsPlusSampleAsymmetry-GaAsOnlyAsymmetry),5)))













