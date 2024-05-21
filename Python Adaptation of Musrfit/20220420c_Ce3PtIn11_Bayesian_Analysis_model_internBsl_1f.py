#%%
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:02:14 2022

@author: sebas
"""
import numpy as np                     # many useful functions in python
import matplotlib.pyplot as plt        # plotting
from iminuit import Minuit             # data fitting : import of the Minuit object
from iminuit.cost import LeastSquares  # function to minimize error
import scipy.special as special        # for bessel or other complex functions

# Bayesian Analysis packages
import emcee
import corner
import dynesty
from dynesty import NestedSampler

#%% custom runplot partagé par Thomas Vandal
import warnings
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
from dynesty.utils import resample_equal as resample_equal

str_type = str
float_type = float
int_type = int

def _make_subplots(fig, nx, ny, xsize, ysize):
    # Setting up default plot layout.
    if fig is None:
        fig, axes = pl.subplots(nx, ny, figsize=(xsize, ysize))
        axes = np.asarray(axes).reshape(nx, ny)
    else:
        fig, axes = fig
        try:
            axes = np.asarray(axes).reshape(nx, ny)
        except ValueError:
            raise ValueError("Provided axes do not match the required shape")
    return fig, axes


def runplot(results,
            span=None,
            logplot=False,
            kde=True,
            nkde=1000,
            color='blue',
            plot_kwargs=None,
            label_kwargs=None,
            lnz_error=True,
            lnz_truth=None,
            truth_color='red',
            truth_kwargs=None,
            max_x_ticks=8,
            max_y_ticks=3,
            use_math_text=True,
            mark_final_live=True,
            fig=None):
    """
    Plot live points, ln(likelihood), ln(weight), and ln(evidence)
    as a function of ln(prior volume).
​
    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run.
​
    span : iterable with shape (4,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds *or* a float from `(0., 1.]` giving the
        fraction below the maximum. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::
​
            span = [(0., 10.), 0.001, 0.2, (5., 6.)]
​
        Default is `(0., 1.05 * max(data))` for each element.
​
    logplot : bool, optional
        Whether to plot the evidence on a log scale. Default is `False`.
​
    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.
​
    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.
​
    color : str or iterable with shape (4,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the lines in each subplot.
        Default is `'blue'`.
​
    plot_kwargs : dict, optional
        Extra keyword arguments that will be passed to `plot`.
​
    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.
​
    lnz_error : bool, optional
        Whether to plot the 1, 2, and 3-sigma approximate error bars
        derived from the ln(evidence) error approximation over the course
        of the run. Default is `True`.
​
    lnz_truth : float, optional
        A reference value for the evidence that will be overplotted on the
        evidence subplot if provided.
​
    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color used when plotting :data:`lnz_truth`.
        Default is `'red'`.
​
    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting
        :data:`lnz_truth`.
​
    max_x_ticks : int, optional
        Maximum number of ticks allowed for the x axis. Default is `8`.
​
    max_y_ticks : int, optional
        Maximum number of ticks allowed for the y axis. Default is `4`.
​
    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.
​
    mark_final_live : bool, optional
        Whether to indicate the final addition of recycled live points
        (if they were added to the resulting samples) using
        a dashed vertical line. Default is `True`.
​
    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the run onto the provided figure.
        Otherwise, by default an internal figure is generated.
​
    Returns
    -------
    runplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output summary plot.
​
    """

    # Initialize values.
    if label_kwargs is None:
        label_kwargs = dict()
    if plot_kwargs is None:
        plot_kwargs = dict()
    if truth_kwargs is None:
        truth_kwargs = dict()

    # Set defaults.
    plot_kwargs['linewidth'] = plot_kwargs.get('linewidth', 5)
    plot_kwargs['alpha'] = plot_kwargs.get('alpha', 0.7)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 3)

    # Extract results.
    niter = results['niter']  # number of iterations
    logvol = results['logvol']  # ln(prior volume)
    logl = results['logl'] - max(results['logl'])  # ln(normalized likelihood)
    logwt = results['logwt'] - results['logz'][-1]  # ln(importance weight)
    logz = results['logz']  # ln(evidence)
    logzerr = results['logzerr']  # error in ln(evidence)
    logzerr[~np.isfinite(logzerr)] = 0.
    nsamps = len(logwt)  # number of samples

    # Check whether the run was "static" or "dynamic".
    try:
        nlive = results['samples_n']
        mark_final_live = False
    except KeyError:
        nlive = np.ones(niter) * results['nlive']
        if nsamps - niter == results['nlive']:
            nlive_final = np.arange(1, results['nlive'] + 1)[::-1]
            nlive = np.append(nlive, nlive_final)

    # Check if the final set of live points were added to the results.
    if mark_final_live:
        if nsamps - niter == results['nlive']:
            live_idx = niter
        else:
            warnings.warn("The number of iterations and samples differ "
                          "by an amount that isn't the number of final "
                          "live points. `mark_final_live` has been disabled.")
            mark_final_live = False

    # Determine plotting bounds for each subplot.
    # TODO: Catch warning and raise error before matplotlib
    data = [nlive, np.exp(logl), np.exp(logwt), logz if logplot else np.exp(logz)]
    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, data[2]))  # KDE
        logvol_new = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        data[2] = wt_kde.pdf(-logvol_new)  # evaluate KDE PDF
    if span is None:
        span = [(0., 1.05 * max(d)) for d in data]
        no_span = True
    else:
        no_span = False
    span = list(span)
    if len(span) != 4:
        raise ValueError("More bounds provided in `span` than subplots!")
    for i, _ in enumerate(span):
        try:
            ymin, ymax = span[i]
        except:
            span[i] = (max(data[i]) * span[i], max(data[i]))
    if lnz_error and no_span:
        if logplot:
            # Same lower bound as in ultranest: https://github.com/JohannesBuchner/UltraNest/blob/master/ultranest/plot.py#L139.
            zspan = (logz[-1] - 10.3 * 3. * logzerr[-1],
                     logz[-1] + 1.3 * 3. * logzerr[-1])
        else:
            zspan = (0., 1.05 * np.exp(logz[-1] + 3. * logzerr[-1]))
        span[3] = zspan

    # Setting up default plot layout.
    had_fig = fig or False
    fig, axes = _make_subplots(fig, 4, 1, 16, 16)
    axes = axes.flatten()
    xspan = [ax.get_xlim() for ax in axes]
    if had_fig:
        yspan = [ax.get_ylim() for ax in axes]
    else:
        yspan = span
    # One exception: if the bounds are the plotting default `(0., 1.)`,
    # overwrite them.
    xspan = [t if t != (0., 1.) else (0., -min(logvol)) for t in xspan]
    yspan = [t if t != (0., 1.) else (None, None) for t in yspan]

    # Set up bounds for plotting.
    for i in range(4):
        if xspan[i][0] is None:
            xmin = None
        else:
            xmin = min(0., xspan[i][0])
        if xspan[i][1] is None:
            xmax = -min(logvol)
        else:
            xmax = max(-min(logvol), xspan[i][1])
        if yspan[i][0] is None:
            ymin = None
        else:
            ymin = min(span[i][0], yspan[i][0])
        if yspan[i][1] is None:
            ymax = span[i][1]
        else:
            ymax = max(span[i][1], yspan[i][1])
        axes[i].set_xlim([xmin, xmax])
        axes[i].set_ylim([ymin, ymax])

    # Plotting.
    labels = [
        'Live Points', 'Likelihood\n(normalized)', 'Importance\nWeight',
        'log(Evidence)' if logplot else 'Evidence'
    ]
    if kde:
        labels[2] += ' PDF'

    for i, d in enumerate(data):

        # Establish axes.
        ax = axes[i]
        # Set color(s)/colormap(s).
        if isinstance(color, str_type):
            c = color
        else:
            c = color[i]
        # Setup axes.
        if max_x_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_x_ticks))
        if max_y_ticks == 0:
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(max_y_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Plot run.
        if logplot and i == 3:
            ax.plot(-logvol, d, color=c, **plot_kwargs)
            yspan = [ax.get_ylim() for _ax in axes]
        elif kde and i == 2:
            ax.plot(-logvol_new, d, color=c, **plot_kwargs)
        else:
            ax.plot(-logvol, d, color=c, **plot_kwargs)
        if i == 3 and lnz_error:
            if logplot:
                # Same mask as in ultranest: https://github.com/JohannesBuchner/UltraNest/blob/master/ultranest/plot.py#L139.
                mask = logz >= ax.get_ylim()[0] - 10
                [
                    ax.fill_between(
                        -logvol[mask],
                        (logz + s * logzerr)[mask],
                        (logz - s * logzerr)[mask],
                        color=c,
                        alpha=0.2) for s in range(1, 4)
                 ]
            else:
                [
                    ax.fill_between(-logvol,
                        np.exp(logz + s * logzerr),
                        np.exp(logz - s * logzerr),
                        color=c,
                        alpha=0.2) for s in range(1, 4)
                ]
        # Mark addition of final live points.
        if mark_final_live:
            ax.axvline(-logvol[live_idx],
                       color=c,
                       ls="dashed",
                       lw=2,
                       **plot_kwargs)
            if i == 0:
                ax.axhline(live_idx, color=c, ls="dashed", lw=2, **plot_kwargs)
        # Add truth value(s).
        if i == 3 and lnz_truth is not None:
            if logplot:
                ax.axhline(lnz_truth, color=truth_color, **truth_kwargs)
            else:
                ax.axhline(np.exp(lnz_truth), color=truth_color, **truth_kwargs)

    return fig, axes
#%% initialisation des variables globales
np.random.seed(0)

tstep = 0.390625*10**-9  # (seconds) time interval of binned data
tstep *= 10**6    # (microseconds)
# set gamma
gamma=0.0135528*10**6
# packing
binSize = 100
# starting point of data to analyze
t0 = 1031
# background to analyze is in the interval [zero,t0-bad]
bad = 100
zero = 75

# cut off data after set number of microseconds
cut = 6 # microseconds, or 6 *10**-6 seconds

alpha = 1.03 # variable based on model

#%% musr functions
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

#%% bayesian analysis functions
def log_prior(vector):
    b1, f1, ph, sg, ll, lt = vector
    
    if (b1min<b1max and f1min<f1<f1max and phmin<ph<phmax and sgmin<sg<sgmax and llmin<ll<llmax and ltmin<lt<ltmax):
        return 0
    #log(1)=0
    return -np.inf
    #log (0)=-inf
    
def prior_transform(vector):
    # on convertit ndim chiffres entre 0 et 1 en ndim nombres bien distribués sur leur range respectif 
    return np.array([vector[0]*b1range+b1min,vector[1]*f1range+f1min,vector[2]*phrange+phmin,vector[3]*sgrange+sgmin, \
                     vector[4]*llrange+llmin,vector[5]*ltrange+ltmin])
    #return np.array([vector[0]*arange+amin,vector[1]*frange+fmin,vector[2]*prange+pmin,vector[3]*srange+smin])
    
def log_likelihood(vector):
    sigma2 = errorBin ** 2
    return -0.5 * np.sum((binA - model(vector)) ** 2 /  sigma2  )#+ np.log(sigma2))

def log_probability(vector):
    lp = log_prior(vector)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(vector)
    
def model(vector):
    #total_asymmetry, field, phi, sigma = vector
    #return total_asymmetry * np.cos(2*np.pi * gamma * field * 10**-6 * binT + np.pi * phi/180) * np.exp(-sigma*binT)
    

    beta, field, phi, sigma, lambdaL, lambdaT  = vector
    asy1 = 0.07
    asy2 = 0.16
    
    simpleGss=np.exp(-1/2*(sigma*binT)**2)
    besselFunction = special.jv(0,2*np.pi*gamma*field*binT*10**-6+np.pi*phi/180)

    internBsl_signal = (beta*besselFunction*np.exp(-lambdaT*binT)+(1-beta)*np.exp(-lambdaL*binT))

    
    return asy1*internBsl_signal+asy2*simpleGss  

#%%
"""
Notre x est le temps t, notre y (a) est l'asymmétrie et yerr (aerr) la variance de l'asymmétrie.
"""
f,b,l,r = clean("005810.txt")
a = (f*alpha-b)/(f*alpha+b)
aerr = np.var(a)
t = np.arange(len(f))*tstep


binAsymmetry = np.zeros([int(np.round((len(f)/binSize)))+1])
binAmount = int(np.round(len(a)/binSize))
binRange = np.linspace(0, len(a), binAmount)*tstep
for j in range(binAmount):
    binAsymmetry[j]= np.mean(a[binSize*j:binSize*j+binSize])
binT=binRange
binA=binAsymmetry[:len(binRange)]
errorBin = ((aerr/binSize**(1/2)))

ndim = 6 # vector a 6 composantes

#b1, f1, ph, sg, ll, lt = vector
# range estimations

b1min=0.;b1max=1.  ;b1range=b1max-b1min
f1min=0.;f1max=250.;f1range=f1max-f1min
phmin=0.;phmax=360.;phrange=phmax-phmin
sgmin=0.;sgmax=1.  ;sgrange=sgmax-sgmin
llmin=0.;llmax=1.  ;llrange=llmax-llmin
ltmin=0.;ltmax=1.  ;ltrange=ltmax-ltmin




#%% Étape 3.1: EMCEE

# on essaie de déterminer par essai-erreur les paramètres pour donner un point
# de départ aux marcheurs
n = 10000
#b1, f1, ph, sg, ll, lt = vector
testVector = [0.49,7,73,0.06,0.2,0.0]

plt.figure(figsize=(12,8))
plt.scatter(binT,binA,s=5,color="green")

testy=model(testVector)
plt.plot(binT,testy,".",color="red")
plt.show()
print(testVector)


initb1=(np.random.random(32)-0.5)*b1range/100+testVector[0]
initf1=(np.random.random(32)-0.5)*f1range/100+testVector[1]
initph=(np.random.random(32)-0.5)*phrange/100+testVector[2]
initsg=(np.random.random(32)-0.5)*sgrange/100+testVector[3]
initll=(np.random.random(32)-0.5)*llrange/100+testVector[4]
initlt=(np.random.random(32)-0.5)*ltrange/100+testVector[5]


pos = np.array([initb1,initf1,initph,initsg,initll,initlt]).T


nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability
)

sampler.run_mcmc(pos, n, progress=True);

fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
samples = sampler.get_chain(discard=100,thin=15)
labels = ["b1","f1","ph","sg","ll","lt"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

flat_samples = sampler.get_chain(discard=100,thin=15,flat=True)
corner.corner(flat_samples,labels=labels)
print("MCMC posterior mean:", np.mean(samples))
print("MCMC posterior std:", np.std(samples))

#from IPython.display import display, Math

finalVector = np.array([0.,0.,0.,0.,0.,0.])
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    #display(Math(txt))
    finalVector[i]=mcmc[1]
    print(labels[i],mcmc[1],"±",q[0])
    

plt.show()
plt.figure(figsize=(12,8))
plt.scatter(binT,binA,s=5,color="green")
final=model(finalVector)
plt.plot(binT,final,".",color="red")
plt.show()
print(np.round(finalVector,5))

#######################################################
# dynesty analysis

# "dynamic" nested sampling
SsamplerDyn= dynesty.DynamicNestedSampler(log_likelihood, prior_transform, ndim)
print(SsamplerDyn.citations)
SsamplerDyn.run_nested()
sresults = SsamplerDyn.results
from dynesty import plotting as dyplot
# summary of the run with dynamic sampling
rfig, raxes = dyplot.runplot(sresults,logplot=False)

# plot traces and 1-D marginalized posteriors.
tfig, taxes = dyplot.traceplot(sresults,show_titles=True)

# plot the 2-D marginalized posteriors.
cfig, caxes = dyplot.cornerplot(sresults,show_titles=True)
rfig.tight_layout()
tfig.tight_layout()
cfig.tight_layout()
plt.show()

# comparing cumulative evidence from different models hints at the model probability
cumulative_evidence = sresults["logz"][-1]
print(cumulative_evidence)
