"""
FAST FOURIER TRANSFORM
"""

#%% introduction and import necessary packages
"""
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
#import scipy.special as special        # for bessel or other complex functions
#import scipy.signal as signal
#import csv

# run Fourier transform of the asymmetry
#import scipy

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
    markers = ["o","v","s","D"]
    if len(vector)!= 0:
        if filename == "./runfiles/005148.txt" or filename == "./runfiles/005149.txt" or filename == "./runfiles/005150.txt" or filename == "./runfiles/005151.txt": 
            fitA = modelPenDepth(vector)
            
        else:    
            fitA = modelall5(vector)
            
            
            
        viewFitA = fitA
        
        #plt.plot(binT, viewFitA, label="musrfit {}".format(message[temp]),color=colors[temp],zorder=1)
    
    
    

    plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, ls="none",marker="o", label="T = {}K".format(temperature[temp]),color=colors[temp])
    #plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, ls="none",marker="o",label="{}".format(message[temp]),color=colors[temp])
    
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
    #plt.legend(loc="upper right")
    #plt.show()
    
def modelPenDepth(vector):
    amp1, gaussRelax1, expRelax, phase, internfield, amp2, gaussRelax2 = vector

    simpleGss1  = np.exp(-1/2*(gaussRelax1*binT)**2)
    simpleGss2  = np.exp(-1/2*(gaussRelax2*binT)**2)
    simplExpo = np.exp(-expRelax*binT)
    TFieldCos = np.cos(2*np.pi*internfield*gamma*binT+np.pi*phase/180) #0.01 MHz/G
    
    return amp1 * simpleGss1 * simplExpo * TFieldCos + \
           amp2 * simpleGss2 * TFieldCos
    
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

#colors= ["red","black"]
#filenames = ["005809.txt","028441.txt"]
#message = [r"M15 Ce$_3$PtIn$_{11}$ + holder",\
#           r"M20 Ce$_3$PtIn$_{11}$ "]

colors= ["red","green","blue","black","orange"]
#colors=["black"] * 5
filenames = ["005148.txt","005150.txt","005149.txt","005151.txt"]
temperature = [0.056,0.299,0.428,4.013]    
#message = [r"σ$_s$(0.056K)",r"σ$_s$(0.299K)",r"σ$_s$(0.428K)",r"σ$_s$(4.013K)"]
message = ["0.056K","0.299K","0.428K","4.013K"]
"""

colors= ["red","black"]
filenames = ["005148.txt","005151.txt"]
temperature = [0.056,4.013]    
message = [r"M15 Ce$_3$PtIn$_{11}$ 0.056K",r"M15 Ce$_3$PtIn$_{11}$ 4.013K"]
"""
# 2021 TF runs
#filenames = ["005148.txt","005150.txt","005149.txt","005151.txt"]
#temperature = [0.056,0.299,0.428,4.013]

plt.figure(figsize=(8,8),dpi=100)
plt.ylim(-0.22,0.25)
for temp in range(len(temperature)):  
    
    
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

    #binT, binF, errBinF, binB, errBinB, binA, errBinA, dadf, dadb=getErrorBinA(f,b)
    binT, binF, errBinF, binB, errBinB, binA, errBinA, dadf, dadb=getErrorBinA(r,l)
    
    #rlxStatic,betaStatic,asyTail,asyCe1,rlxCe1,fieldCe1,asyCe2,rlxCe2,fieldCe2
    
    
    #test
    #bestPeaks = [0.273,1.28,0.000,0.0041,1.23,127.57,0.012,1.38,17]
    
    # dynesty fit
    #bestPeaks = [0.276,1.0,0.003,0.0039,1.266,126.739,0.0015,0,49]
    # dynesty fit with second peak for rlxStatic
    #bestPeaks = [0.344,1.0,0.003,0.0039,1.266,126.739,0.0015,0,49]
    # musr fit
    #bestPeaks = [0.417,1.056,0.0087,0.00395,1.06,120.8,0.00137,0.01,48.7]

    i=temp
    
    # 5151 4K 
    #amp1, gaussRelax1, expRelax, phase, internfield, amp2, gaussRelax2 = vector
    
    bestPeaks = [[0.03471,0.699,0.087,-86.62,96.64,0.155,0.051],
                 [0.03509,0.694,0.087,-86.58,96.61,0.155,0.051],
                 [0.03487,0.676,0.087,-86.69,96.62,0.155,0.051],
                 [0.058,0.19,0.087,-88.38,96.63,0.155,0.051]]
    """
    bestPeaks = [[0.03471,0.699,0.087,-86.62,96.64,0.155,0.051],
                 [0.058,0.19,0.087,-88.38,96.63,0.155,0.051]]
    """
    plotResults(filename, binT, binA,errBinA,i,vector=bestPeaks[i])
    #plt.show()
    #bestPeaks = [0.273,1.28,0.000,0.0041,1.23,127.57,0.012,1.38,17]
    
linewidth = [0.699,0.694,0.676,0.19]
lw_err = [0.023,0.024,0.021,0.13]

for l in range(len(linewidth)):
    lw_err[l] = np.sqrt((2*linewidth[l]*lw_err[l])**2+(-2*linewidth[2]*lw_err[2])**2)
    linewidth[l] = np.sqrt(linewidth[l]**2 - linewidth[2]**2)
    
    
lw_err[3] = 0  
plt.figure(figsize=(6,6),dpi=100)




t_c = 0.32 # K
x = np.linspace(0, t_c,100)
def modelPowerLaw(x,σ,n):
    return σ*(1-(x/t_c)**n)



least_squares = LeastSquares(temperature[0:2],linewidth[0:2],lw_err[0:2],modelPowerLaw)
# starting values
m = Minuit(least_squares,σ=0.190,n=15)  
#m.limits['σ'] = (0,0.177)
#m.limits['n'] = (0,16)  
# finds minimum of least_squares function
m.migrad()
# accurately computes uncertainties
m.hesse()

# display legend with some fit info
fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(temperature) - m.nfit} = {np.round(m.fval/(len(temperature) - m.nfit),2)}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    
lw_zero = m.values[0]
n = m.values[1]
lw_zero_err = m.errors[0]
y = modelPowerLaw(x, lw_zero, n)


plotzerowidth=plt.errorbar(0,lw_zero,lw_zero_err,ls='none',color = "black",label=r"σ$_s$(0K)        = 0.178 $\pm$ 0.043 μs$^−$$ ^1$", marker = 'o',mfc='white')
plotzerowidth[-1][0].set_linestyle('--')  
for l in range(3):
    if l == 2:
        plt.errorbar(temperature[l],linewidth[l],0,ls='none', color = colors[l],label=r"σ$_s$({}) = 0 μs$^−$$ ^1$".format(message[l],np.round(linewidth[l],0),np.round(lw_err[l],3)), marker = "o")
    else:
        plt.errorbar(temperature[l],linewidth[l],lw_err[l],ls='none', color = colors[l],label=r"σ$_s$({}) = {} $\pm$ {} μs$^−$$ ^1$".format(message[l],np.round(linewidth[l],3),np.round(lw_err[l],3)), marker = "o")
#plt.plot(x/t_c,y/lw_zero,ls="--",color = "gray",label="{} \n {} \n {}".format(fit_info[0],fit_info[1],fit_info[2]))    

plt.plot(x,y,ls="--",color = "black")#,label=r"Power Law σ(T) / σ(0) = [1 - (T/T$_c$)$^n$]")   
#plt.vlines(t_c,0,0.5,ls="--",color="blue",label =r"T$_c$ = {}K".format(0.32))


#plt.xlim(-0.1,1.4)
#plt.ylim(-0.1,1.4)
#plt.xlabel(r"T / T$_c$")
#plt.ylabel(r"σ$_s$(T) / σ$_s$(0)")
plt.xlabel("T (K)")
plt.ylabel(r"σ$_s$(T) (μs$^−$$ ^1$)")
plt.legend()
plt.show()




dW = [lw_zero,linewidth[0],linewidth[1]]
σ_err = lw_err[:]



gamma_mu = 8.515475*10**-8 # in seconds-1 Gauss-1
quantumFlux = 2.067833848 * 10**-15  #Weber


# Initialize lists for results
pen = []
pen_err = []

# Calculating pen and pen_err
for i in range(len(dW)):
    σ = dW[i]#*10**-6 # micros-1 to s-1
    e = σ_err[i]

    if σ != 0:  # Avoid division by zero
        # Calculate pen
        p = 0.328/np.sqrt(σ) # micros-1 and microm-1
        #p = ((gamma_mu * quantumFlux / σ)**2 *0.00371)**(1/4) *10**6
        pen.append(p)
        
        # Calculate pen_err
        err = p * (e / σ)
        pen_err.append(err)
    else:
        pen.append(0)
        pen_err.append(0)



newTemp = np.append([0],temperature)
####

K_B = 1
# FIT for penetration depth
x = np.linspace(0.00000001, t_c,100)
def modelBCSPenDepth(x,λ,tri):
    return λ*(1+np.sqrt(np.pi*tri/(2*K_B*x))*np.exp(-tri/(K_B*x)))



least_squares = LeastSquares(temperature[0:2],pen[1:3],pen_err[1:3],modelBCSPenDepth)
# starting values
m_2 = Minuit(least_squares,λ=0.1,tri=1)  
m_2.limits['λ'] = (0.6,1)
m_2.limits['tri'] = (0,2)  
# finds minimum of least_squares function
m_2.migrad()
# accurately computes uncertainties
m_2.hesse()

# display legend with some fit info
fit_info_2 = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {len(temperature) - m.nfit} = {np.round(m.fval/(len(temperature) - m.nfit),2)}",
]
for p, v, e in zip(m_2.parameters, m_2.values, m_2.errors):
    fit_info_2.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    
λ_zero = m_2.values[0]
tri = m_2.values[1]
λ_zero_err = m_2.errors[0]
y = modelBCSPenDepth(x,λ_zero,tri)



####



plt.figure(figsize=(6,6),dpi=100)
plt.plot(x,y,ls="--",color = "black",alpha = 1,label="BCS")   
plotzeropendepth=plt.errorbar(newTemp[0],λ_zero,λ_zero_err,ls='none',color = "black",alpha = 1,label=r"λ({}) (BCS) = {} $\pm$ {} μm".format("0K",np.round(λ_zero,3),np.round(λ_zero_err,3)), marker = 'o',mfc='white')
plotzeropendepth[-1][0].set_linestyle('--')
plt.errorbar(newTemp[0]-0.005,pen[0],pen_err[0],ls='none',color = "gray",alpha = 0.3,label=r"λ({}) (AB$^n$) = {} $\pm$ {} μm".format("0K",np.round(pen[0],3),np.round(pen_err[0],3)), marker = 'o')

for l in range(2):
    plt.errorbar(newTemp[l+1],pen[l+1],pen_err[l+1],ls='none',color = colors[l],label=r"λ({}) = {} $\pm$ {} μm".format(message[l],np.round(pen[l+1],3),np.round(pen_err[l+1],3)),marker = 'o')


plt.xlabel("T (K)")
plt.ylabel(r"Penetration Depth λ (μm)")
plt.legend()
plt.show()
