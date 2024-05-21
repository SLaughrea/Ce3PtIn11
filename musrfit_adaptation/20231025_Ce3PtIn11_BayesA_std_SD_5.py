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

# Bayesian Analysis packages
import emcee
#import corner
import dynesty
from dynesty import NestedSampler
from dynesty import utils as dyfunc

# run Fourier transform of the asymmetry
import scipy
#%% custom runplot by Thomas Vandal
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
#%% custom traceplot

# import needed functions
import logging
import warnings
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.ndimage import gaussian_filter as norm_kde
from scipy.stats import gaussian_kde
from dynesty.utils import resample_equal, unitcheck
from dynesty.utils import quantile as _quantile
#from dynesty.utils import get_random_generator, get_nonbounded
from dynesty import bounding

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


def rotate_ticks(ax, xy):
    if xy == 'x':
        labs = ax.get_xticklabels()
    else:
        labs = ax.get_yticklabels()
    for lab in labs:
        lab.set_rotation(45)


def plot_thruth(ax,
                truths,
                truth_color,
                truth_kwargs,
                vertical=None,
                horizontal=None):
    """
Plot the thruth line (horizontal or vertical).
truths can be None or one value or a list
"""
    if vertical:
        func = ax.axvline
    elif horizontal:
        func = ax.axhline
    else:
        raise ValueError('vertical or horizontal option must be specified')
    if truths is not None:
        try:
            curt = iter(truths)
        except TypeError:
            curt = [truths]
        for t in curt:
            func(t, color=truth_color, **truth_kwargs)


def check_span(span, samples, weights):
    """
If span is a list of scalars, replace it by the list of bounds.
If the input is list of pairs, it is kept intact
    """
    for i, _ in enumerate(span):
        try:
            iter(span[i])
            if len(span[i]) != 2:
                raise ValueError('Incorrect span value')
        except TypeError:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = _quantile(samples[i], q, weights=weights)


def traceplot(results,
              span=None,
              quantiles=(0.025, 0.5, 0.975),
              smooth=0.02,
              thin=1,
              dims=None,
              post_color='blue',
              post_kwargs=None,
              kde=True,
              nkde=1000,
              trace_cmap='plasma',
              trace_color=None,
              trace_kwargs=None,
              connect=False,
              connect_highlight=10,
              connect_color='red',
              connect_kwargs=None,
              max_n_ticks=5,
              use_math_text=False,
              labels=None,
              label_kwargs=None,
              show_titles=False,
              title_quantiles=(0.025, 0.5, 0.975),
              title_fmt=".2f",
              title_kwargs=None,
              truths=None,
              truth_color='red',
              truth_kwargs=None,
              verbose=False,
              fig=None):
    """
    Plot traces and marginalized posteriors for each parameter.
    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.
    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::
            span = [(0., 10.), 0.95, (5., 6.)]
        Default is `0.999999426697` (5-sigma credible interval) for each
        parameter.
    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (the 95%/2-sigma credible interval).
    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.
    thin : int, optional
        Thin the samples so that only each `thin`-th sample is plotted.
        Default is `1` (no thinning).
    dims : iterable of shape (ndim,), optional
        The subset of dimensions that should be plotted. If not provided,
        all dimensions will be shown.
    post_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'blue'`.
    post_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the
        marginalized 1-D posteriors.
    kde : bool, optional
        Whether to use kernel density estimation to estimate and plot
        the PDF of the importance weights as a function of log-volume
        (as opposed to the importance weights themselves). Default is
        `True`.
    nkde : int, optional
        The number of grid points used when plotting the kernel density
        estimate. Default is `1000`.
    trace_cmap : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style colormap (either a single colormap or a
        different colormap for each subplot) used when plotting the traces,
        where each point is colored according to its weight. Default is
        `'plasma'`.
    trace_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a
        different color for each subplot) used when plotting the traces.
        This overrides the `trace_cmap` option by giving all points
        the same color. Default is `None` (not used).
    trace_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the traces.
    connect : bool, optional
        Whether to draw lines connecting the paths of unique particles.
        Default is `False`.
    connect_highlight : int or iterable, optional
        If `connect=True`, highlights the paths of a specific set of
        particles. If an integer is passed, :data:`connect_highlight`
        random particle paths will be highlighted. If an iterable is passed,
        then the particle paths corresponding to the provided indices
        will be highlighted.
    connect_color : str, optional
        The color of the highlighted particle paths. Default is `'red'`.
    connect_kwargs : dict, optional
        Extra keyword arguments used for plotting particle paths.
    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.
    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.
    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.
    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.
    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the 0.5 quantile along with the upper/lower bounds associated
        with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
        Default is `False`.
    title_quantiles : iterable, optional
        A list of fractional quantiles to use in the title. Default is
        `[0.025, 0.5, 0.975]` (median plus 95%/2-sigma credible interval).
    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.
    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.
    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.
    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.
    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.
    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.
    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.
    Returns
    -------
    traceplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output trace plot.
    """
    
    # Initialize values.
    if title_kwargs is None:
        title_kwargs = {}
    if label_kwargs is None:
        label_kwargs = {}
    if trace_kwargs is None:
        trace_kwargs = {}
    if connect_kwargs is None:
        connect_kwargs = {}
    if post_kwargs is None:
        post_kwargs = {}
    if truth_kwargs is None:
        truth_kwargs = {}

    # Set defaults.
    connect_kwargs['alpha'] = connect_kwargs.get('alpha', 0.7)
    post_kwargs['alpha'] = post_kwargs.get('alpha', 0.6)
    trace_kwargs['s'] = trace_kwargs.get('s', 3)
    trace_kwargs['edgecolor'] = trace_kwargs.get('edgecolor', None)
    trace_kwargs['edgecolors'] = trace_kwargs.get('edgecolors', None)
    truth_kwargs['linestyle'] = truth_kwargs.get('linestyle', 'solid')
    truth_kwargs['linewidth'] = truth_kwargs.get('linewidth', 2)
    #rstate = get_random_generator()
    # Extract weighted samples.
    samples = results['samples']
    logvol = results['logvol']
    weights = np.exp(results['logwt'] - results['logz'][-1])

    if kde:
        # Derive kernel density estimate.
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], nkde)  # resample
        wt_grid = wt_kde.pdf(-logvol_grid)  # evaluate KDE PDF
        wts = np.interp(-logvol, -logvol_grid, wt_grid)  # interpolate
    else:
        wts = weights

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"

    # Slice samples based on provided `dims`.
    if dims is not None:
        samples = samples[dims]
    ndim, nsamps = samples.shape
    

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Check ln(volume).
    if logvol.ndim != 1:
        raise ValueError("Ln(volume)'s must be 1-D.")
    if nsamps != logvol.shape[0]:
        raise ValueError("The number of ln(volume)'s and samples disagree!")
    
    # Determine plotting bounds for marginalized 1-D posteriors.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    check_span(span, samples, weights)
    
    # Setting up labels.
    if labels is None:
        labels = [r"$x_{" + str(i + 1) + "}$" for i in range(ndim)]

    # Setting up smoothing.
    if isinstance(smooth, (int_type, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setting up default plot layout.
    fig, axes = _make_subplots(fig, ndim, 2, 12, 3 * ndim)

    posterior_pdf = [None]*len(samples)
    # Plotting.
    for i, x in enumerate(samples):
        
        # Plot trace.

        # Establish axes.
        ax = axes[i, 0]
        # Set color(s)/colormap(s).
        if trace_color is not None:
            if isinstance(trace_color, str_type):
                color = trace_color
            else:
                color = trace_color[i]
        else:
            color = wts[::thin]
        if isinstance(trace_cmap, str_type):
            cmap = trace_cmap
        else:
            cmap = trace_cmap[i]
        # Setup axes.
        ax.set_xlim([0., -min(logvol)])
        ax.set_ylim([min(x), max(x)])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks))
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.yaxis.set_major_formatter(sf)
        ax.set_xlabel(r"$-\ln X$", **label_kwargs)
        ax.set_ylabel(labels[i], **label_kwargs)
        # Generate scatter plot.
        ax.scatter(-logvol[::thin],
                   x[::thin],
                   c=color,
                   cmap=cmap,
                   **trace_kwargs)
        if connect:
            # Add lines highlighting specific particle paths.
            for j in ids:
                sel = (samples_id[::thin] == j)
                ax.plot(-logvol[::thin][sel],
                        x[::thin][sel],
                        color=connect_color,
                        **connect_kwargs)
        # Add truth value(s).
        if truths is not None:
            plot_thruth(ax,
                        truths[i],
                        truth_color,
                        truth_kwargs,
                        horizontal=True)

        # Plot marginalized 1-D posterior.

        ax = axes[i, 1]
        # Set color(s).
        if isinstance(post_color, str_type):
            color = post_color
        else:
            color = post_color[i]
        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        ax.set_xlabel(labels[i], **label_kwargs)
        # Generate distribution.
        s = smooth[i]
        if isinstance(s, int_type):
            # If `s` is an integer, plot a weighted histogram with
            # `s` bins within the provided bounds.
            n, b, _ = ax.hist(x,
                              bins=s,
                              weights=weights,
                              color=color,
                              range=np.sort(span[i]),
                              **post_kwargs)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / s))
            n, b = np.histogram(x,
                                bins=bins,
                                weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            x0 = 0.5 * (b[1:] + b[:-1])
            y0 = n

            posterior_pdf[i] = n,b#plt.fill_between(x0, y0, color=color, **post_kwargs)
            ax.fill_between(x0, y0, color=color, **post_kwargs)
        ax.set_ylim([0., max(y0) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = _quantile(x, quantiles, weights=weights)
            for q in qs:
                ax.axvline(q, lw=2, ls="dashed", color=color)
            if verbose:
                print("Quantiles:")
                print(labels[i], list(zip(quantiles, qs)))
        # Add truth value(s).
        if truths is not None:
            plot_thruth(ax,
                        truths[i],
                        truth_color,
                        truth_kwargs,
                        vertical=True)
        # Set titles.
        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = _quantile(x, title_quantiles, weights=weights)
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))
                title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

    return fig, axes, posterior_pdf
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

def getErrorBinA(f,b):
    
    binAmount = int(np.round(len(f)/binSize))
    binT = np.linspace(0, len(f), binAmount)*tstep
    # initialize bins of f, b and a
    binB = np.zeros_like(binT)
    binF = np.zeros_like(binT)
    binA = np.zeros_like(binT)
    # initialize error bins for f, b and a
    errB = np.zeros_like(binB)
    errF = np.zeros_like(binF)
    errA = np.zeros_like(binA)
    
    for i in range(binAmount):
    # calculate bins of f and b
        
        B = b[binSize*i:binSize*i+binSize]
        F = f[binSize*i:binSize*i+binSize]
        
        binB[i] = np.mean(b[binSize*i:binSize*i+binSize])
        binF[i] = np.mean(f[binSize*i:binSize*i+binSize])
        
        #varB = np.mean([((x - binB[i]) ** 2) for x in B])
        #varF = np.mean([((x - binF[i]) ** 2) for x in F])
        
        varB = np.var(B)
        varF = np.var(F)


        errB[i] = np.sqrt(1+varB)#/binB[i]
        errF[i] = np.sqrt(1+varF)#/binF[i]
        
    
    # calculate partial derivative of asymmetry by f and by b
    dadf = 2*alpha*binB/(binF*alpha+binB)**2
    dadb =-2*alpha*binF/(binF*alpha+binB)**2
    
    # propagate standard error 
    #df(x,y,...) = sqrt( (df/dx *dx)**2 + (df/dy * dy)**2 + ...)  
    
    
    # musrfit formula http://lmu.web.psi.ch/musrfit/user/html/user-manual.html#fit-types
    errA = np.sqrt((dadf*errF)**2+(dadb*errB)**2)
    binA = (alpha*(binF)-(binB))/(alpha*beta*(binF)+(binB))
    """
    # Jeff Sonier formula https://musr.ca/theses/Sonier/MSc/node26.html
    binA = ((1-alpha)+(binF-binB)/(binF+binB)*(1+alpha*beta))/     \
           ((1+alpha)+(binF-binB)/(binF+binB)*(1-alpha*beta))
    
    dAdF =  (4*alpha*binB*(beta+1))/(alpha*binF+alpha*binB-alpha*beta*binF+alpha*beta*binB+2*binF)**2
    dAdB = -(4*alpha*binF*(beta+1))/(alpha*binF+alpha*binB-alpha*beta*binF+alpha*beta*binB+2*binF)**2
    dAdalpha = 0 # doesnt matter since alpha uncertainty = 0
    dAdbeta = 0  # doesnt matter since beta uncertainty = 0
    
    #ARE THEY 0??? WE HAVE AN UNCERTAINTY
    erralpha = 0
    errbeta = 0
    errA = np.sqrt((dAdF * errF)**2 + (dAdB * errB)**2 + (dAdalpha * erralpha)**2 + (dAdbeta *errbeta)**2)
    """

    # return error of asymmetry bins
    return binT, binF, errF, binB, errB, binA, errA

def plotResults(filename,goodBinT,goodBinA,goodErrorBinA,vector):
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
    plt.plot(binT, modelall5(vector), label="fit",color="orange")
    plt.errorbar(viewGoodBinT, viewGoodBinA, viewGoodErrorBinA, fmt=".", label="data",color="deepskyblue")
    title = "Asymmetry fit for run " + filename + " with all5"
    plt.xlabel("time (s)",fontsize=12)
    plt.ylabel("Asymmetry",fontsize=12)        
    plt.title(title,fontsize=12)
    #plt.legend(title="$\\chi^2$ / $n_\\mathrm{{dof}}$ = {0}/{1} = {2}".format(chi2,(N-1),chi2dof),fontsize=12,title_fontsize=12)
    plt.legend()
    plt.show()
#%% bayesian analysis functions
def log_prior(vector):

    if (ranges[0][0] < vector[0] < ranges[0][1] and\
        ranges[1][0] < vector[1] < ranges[1][1] and\
        ranges[2][0] < vector[2] < ranges[2][1] and\
        ranges[3][0] < vector[3] < ranges[3][1] and\
        ranges[4][0] < vector[4] < ranges[4][1] and\
        ranges[5][0] < vector[5] < ranges[5][1] and\
        ranges[6][0] < vector[6] < ranges[6][1] and\
        ranges[7][0] < vector[7] < ranges[7][1] and\
        ranges[8][0] < vector[8] < ranges[8][1] ):
        return 0
    return -np.inf
    
def prior_transform(vector):
    # convert ndim numbers between 0 and 1 to ndim numbers distributed on their respective range
    return np.array([vector[0]*ranges[0][2]+ranges[0][0],\
                     vector[1]*ranges[1][2]+ranges[1][0],\
                     vector[2]*ranges[2][2]+ranges[2][0],\
                     vector[3]*ranges[3][2]+ranges[3][0],\
                     vector[4]*ranges[4][2]+ranges[4][0],\
                     vector[5]*ranges[5][2]+ranges[5][0],\
                     vector[6]*ranges[6][2]+ranges[6][0],\
                     vector[7]*ranges[7][2]+ranges[7][0],\
                     vector[8]*ranges[8][2]+ranges[8][0]])
        
def log_likelihood_all5(vector):
    sigma2 = errA ** 2
    return -0.5 * np.sum((binA - modelall5(vector)) ** 2 /  sigma2  )

def log_probability_all5(vector):
    lp = log_prior(vector)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_all5(vector)
    
def modelall5(vector):
    rlxStatic,betaStatic,asyTail,asyCe1,rlxCe1,fieldCe1,asyCe2,rlxCe2,fieldCe2 = vector
    # background + static field (electronic below TN, nuclear above) + nonrelaxing longtime tail below TN + precession signal 1 + precession signal 2
    simplExpoBkgd = np.exp(-rlxAg*binT)
    generExpo = np.exp(-(rlxStatic*binT)**betaStatic)
    simplExpo1 = np.exp(-rlxCe1*binT)
    simplExpo2 = np.exp(-rlxCe2*binT)
    TFieldCos1 = np.cos(2*np.pi*fieldCe1*gamma*binT)
    TFieldCos2 = np.cos(2*np.pi*fieldCe2*gamma*binT)
    
    asyStatic = asysample - asyTail - asyCe1 - asyCe2
    
    return asybkgd*simplExpoBkgd        + \
           asyStatic*generExpo          + \
           asyTail                      + \
           asyCe1*simplExpo1*TFieldCos1 + \
           asyCe2*simplExpo2*TFieldCos2
    
# plot results and highlight peaks
def posterior_peaks(posterior_pdf,showResults=True):

    peaks_table = []
    peaks_error_table = []
    peaks_rel_height_table = []
    for i in range(len(posterior_pdf)):
        x = posterior_pdf[i][1][:-1]
        y = posterior_pdf[i][0]
        if showResults:
            plt.figure(figsize=(10,10))
        dist2x = x[1]-x[0]

        x = np.append(x,x[-1]+dist2x)
        x = np.append(x[0]-dist2x,x)
        y = np.append(0,y)
        y = np.append(y,0)

        peaks, properties = signal.find_peaks(y,height=0.001)
        if len(peaks) == 0:
            peaks = [0]
        peaks_width = signal.peak_widths(y,peaks,rel_height=0.5)
        peaks_rel_height = properties["peak_heights"]/max(properties["peak_heights"])
        if showResults:
            plt.plot(x[peaks], y[peaks], "X",markersize="2",color="red")
        peaks_table.append(x[peaks])
        peaks_rel_height_table.append(peaks_rel_height)
        peaks_error = []
        legend = ""
        # for each peak, there is an uncertainty, we take FWHM
        for k in range(len(peaks)):
            left_peaks_width = x[int(peaks_width[2][k])]
            right_peaks_width = x[int(peaks_width[3][k])]
            if showResults:
                plt.axvline(x[peaks][k],0,1,linestyle="--",color="red")
                plt.axvline(left_peaks_width,0,1,linestyle="--",color="black",alpha=0.2)
                plt.axvline(right_peaks_width,0,1,linestyle="--",color="black",alpha=0.2)
            #legend += "\nPeak {} = ${{:.3f}^{:.3f}_{:.3f}$".format(k+1,x[peaks][k],,)
            qm = x[peaks][k]
            ql = left_peaks_width
            qh = right_peaks_width
            q_minus, q_plus = qm - ql, qh - qm
            fmt = "{{0:{0}}}".format(".4f").format
            fmt1 = "{{0:{0}}}".format(".3f").format
            legend += "\nPeak {0} = ${{{1}}}_{{-{2}}}^{{+{3}}}$".format((k+1), fmt1(qm), fmt(q_minus), fmt(q_plus))
            peaks_error.append([q_minus,q_plus])
        if showResults:
            plt.plot(x,y,label=legend)
            plt.fill_between(x, y)
        peaks_error_table.append(peaks_error)  
        del peaks_error
        title = "{}".format(labels[i])
        if showResults:
            #xmin = ranges[i][0]
            #xmax = ranges[i][1]
            
            plt.ticklabel_format(axis="x",style="scientific")
            #plt.xlim(xmin,xmax)
            plt.title(title)
            plt.legend(fontsize=12,title_fontsize=12)
            plt.show()
    return peaks_table, peaks_error_table, peaks_rel_height_table

def saveFile(filename,array):
    filename = filename + ".npy"
    #array = np.array(array, dtype = object)
    np.save(filename,array, allow_pickle = True)

# open save file instead of recalculating
# option to create temporary files to avoid overwriting
def openSingleFile(filename):
    filename = filename + ".npy"
    array = np.load(filename, allow_pickle = True)
    return array
#%% global variable initialization

"""
START LOOP OVER FILES HERE
"""
# 2020 ZF
#filenames = ["005813.txt","005812.txt","005816.txt","005811.txt","005810.txt","005809.txt"]
#temperature = [0.019,0.228,0.423,0.999,2.05,4.012]

# 2021 ZF
#filenames = ["005138.txt","005137.txt","005136.txt","005142.txt","005139.txt","005141.txt","005140.txt"]
#temperature = [0.056,0.115,0.228,0.427,1.195,1.497,1.799]

# 2020 + 2021 ZF
filenames = ["005813.txt","005138.txt","005137.txt","005136.txt","005812.txt","005816.txt","005142.txt","005811.txt","005139.txt","005141.txt","005140.txt","005810.txt","005809.txt"]
temperature = [0.019,0.056,0.115,0.228,0.228,0.423,0.427,0.999,1.195,1.497,1.799,2.05,4.012]

# 2020 LF 1kG runs
#filenames = ["005820.txt","005819.txt","005818.txt","005817.txt"]
#temperature = [0.02,1.000,2.05,4.011]


bestPeaksTable = [] 
errBestPeaksTable = []
for filename in filenames:    
    
    np.random.seed(0)
    tstep = 0.390625*10**-9  # (seconds) time interval of binned data
    tstep *= 10**6    # (microseconds)
    # set gamma
    gamma=0.0135528*10**6
    # packing
    binSize =25
    default = 200
    # starting point of data to analyze
    t0 = 1031
    # background to analyze is in the interval [zero,t0-bad]
    bad = 100
    zero = 75
    # cut off data after set number of microseconds
    cut = 6 # microseconds, or 6 *10**-6 seconds
    
    
    
    
    if filename == "005813.txt" or filename == "005812.txt" or filename == "005816.txt"\
    or filename == "005811.txt" or filename == "005810.txt" or filename == "005809.txt"\
    or filename == "005820.txt" or filename == "005819.txt" or filename == "005818.txt" or filename == "005817.txt":
        alpha = 1.0746 # 5808_srd.msr
        beta = 1.259 # 5808_h12.msr called relasy
        total_asymmetry = 0.06371 # 5808_srd.msr
        asysample = total_asymmetry
        asybkgd = 0.13877 # 5807.msr
        rlxAg = 0.0091 #rlxAg found with 5808_srd.msr

        
    if filename == "005138.txt" or filename == "005137.txt" or filename == "005136.txt" or filename == "005142.txt"\
    or filename == "005139.txt" or filename == "005141.txt" or filename == "005140.txt":
        alpha = 0.9632 # 5144_srd.msr
        beta = 1.2276 # 5144_h12.msr ?
        total_asymmetry = 0.05355 # 5144_srd.msr
        asysample = total_asymmetry
        asybkgd = 0.1495 # 5131.msr
        rlxAg = 0.0091 #rlxAg found with 5808_srd.msr

    
    #filename = "005813.txt"
    f,b,l,r = clean(filename)
    binT, binF, errF, binB, errB, binA, errA=getErrorBinA(f,b)
    """
    plt.plot(binT,binA)
    plt.show()
    fftResults = scipy.fft.fft(binA)
    newFFT = scipy.fft.fftshift(fftResults)
    n = np.arange(0,len(binT),1)
    plt.plot(1/(tstep*n),newFFT)
    plt.xlim(0,100)
    """
    """
    # Perform FFT on binA
    fft_result = np.fft.fft(binA)
    
    # Calculate the corresponding frequencies in Hz
    sampling_rate = 1 / (binT[1] - binT[0])  # Assuming binT is uniformly sampled
    duration = (binT[1] - binT[0])
    frequencies = np.fft.fftfreq(len(binA), duration) * sampling_rate / 1e6
    
    # Plot the FFT result in frequency (Hz) units
    plt.plot(frequencies, np.abs(fft_result))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT Result')
    plt.show()
    """
    # also return yerr on bins
    
    #errorBinA=getErrorBinA(f,b,binRange,a,filename,alpha)
    labels = ["rlxStatic", "betaStatic", "asyTail", "asyCe1",\
              "rlxCe1", "fieldCe1", "asyCe2", "rlxCe2","fieldCe2"]
    ndim = len(labels)
    ranges = np.full(ndim,None)
    # label[i] = [min, max, range]
    ranges[0] = [0.2,0.8,0.6]
    ranges[1] = [1,2,1]
    ranges[2] = [0.005,0.025,0.02]
    ranges[3] = [0,0.02,0.02]
    ranges[4] = [0.0,2,2]
    ranges[5] = [110,160,50]
    ranges[6] = [0,0.02,0.02]
    ranges[7] = [0.,2,2]
    ranges[8] = [36,66,30]
    
    if filename == "005810.txt" or filename == "005809.txt":
        ranges[3] = [0,0,0]
        ranges[4] = [0,0,0]
        ranges[5] = [0,0,0]
        ranges[6] = [0,0,0]
        ranges[7] = [0,0,0]
        ranges[8] = [0,0,0]
#%% dynesty analysis for both models
    """
    print("Beginning of nested sampling to optimize parameters of the run {}".format(filename))

    # internFld "dynamic" nested sampling
    samplerDyn_all5 = dynesty.DynamicNestedSampler\
        (log_likelihood_all5, prior_transform, ndim=ndim,bound="multi",sample="rslice",bootstrap=50)
    #help(samplerDyn_all5)
    # run with checkpointing
    
    
    samplerDyn_all5.run_nested()
    
    #samplerDyn_all5 = NestedSampler.restore('dynesty.save')
    # resume
    #samplerDyn_all5.run_nested(resume=True)

    sresults_all5 = samplerDyn_all5.results
    #samples_all5, weights_all5 = sresults_all5.samples, np.exp(sresults_all5.logwt - sresults_all5.logz[-1])
    #mean_all5, cov_all5 = dyfunc.mean_and_cov(samples_all5, weights_all5)
    
    # plot traces and 1-D marginalized posteriors.
    tfig, taxes, posterior_pdf_all5 = traceplot(sresults_all5,show_titles=True)
    tfig.tight_layout()
    plt.show()
    
    # comparing cumulative evidence from different models hints at the model probability
    all5_cumulative_evidence = sresults_all5["logz"][-1]
    #print("all5 log_z posterior evidence = {:}".format(all5_cumulative_evidence))
    samples_all5, weights_all5 = sresults_all5.samples, np.exp(sresults_all5.logwt - sresults_all5.logz[-1])
    """
#%% show relevant figures    
    #savename = "save_Amusrfit_bin1_" + filename + "_posterior_pdf"
    #savename = "save_AJeff" + filename + "_posterior_pdf"
    savename = "save_" + filename + "_posterior_pdf"
    posterior_pdf_all5 = openSingleFile(savename)
    #saveFile(savename,posterior_pdf_all5)
    
    # comparing cumulative evidence from different models hints at the model probability
    peaks_table_all5, peaks_error_table_all5, peaks_rel_height_table_all5=posterior_peaks(posterior_pdf_all5,showResults=False)
    
    # get index of peak with highest rel height
    bestPeaks = np.zeros(len(peaks_table_all5))
    errBestPeaks = np.zeros(len(peaks_table_all5))

    for i in range(len(peaks_table_all5)):  
        bestPeaks[i] = peaks_table_all5[i][np.argmax(peaks_table_all5[i])]
        #only save positive error
        errBestPeaks[i] = peaks_error_table_all5[i][np.argmax(peaks_table_all5[i])][1]
    bestPeaksTable = np.append(bestPeaksTable,[bestPeaks])
    bestPeaksTable = np.reshape(bestPeaksTable,(-1,bestPeaks.size))
    errBestPeaksTable = np.append(errBestPeaksTable,[errBestPeaks])
    errBestPeaksTable = np.reshape(errBestPeaksTable,(-1,errBestPeaks.size))
    
    plotResults(filename,binT,binA,errA,bestPeaks)
    
    #fftResults = scipy.fft.ifft(binA)
    #plt.plot(fftResults)
    
    #plt.errorbar(binT,binF,errF)
    #plt.errorbar(binT,binB,errB)
    plt.show()
"""
for i in range(len(labels)):
    fig, ax = plt.subplots(figsize=(8,6))    
    ax.errorbar(temperature,bestPeaksTable[:,i],errBestPeaksTable[:,i],color="black",ecolor="black",marker="o")
    plt.title(labels[i])

    plt.show()
"""
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    