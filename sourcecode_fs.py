import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import rfft, irfft, rfftfreq
from ipywidgets import widgets
from IPython.display import display

# 
# Fourier Series Code
# 

#  funtions for fourier approximation
def f1(x):
    ''' f1(x) = x
    ''' 
    return x

def f2(x):
    ''' Square wave 
               ~ -1, -1 < x < 0
        f2(x)= |
               ~  1,  0 < x < 1
    '''
    return signal.square(2*x)

def f3(x):
    ''' Cubic
    '''
    
    return 5*x*(x+1.)*(x-1.)

def produceFourierSeries(x, nMax, x0, x1, f):
    """ Produces Fourier Series approximation of a function
    
        ----------

        Parameters

        ----------

        x: array_like
            sample values

        nMax: int
            number of terms

        x0: float
            lower bound on window/approximation

        x1: float
            upper bound on window/approximation

        f: function
            function to be approximated

    """
    T = x1 - x0

    # need new range over the actual range of the approximation (not the plotting range)
    # so integration is correct
    xFuncPeriod = np.arange(x0, x1, 0.001)

    series = np.zeros(len(x))
    prev = None

    for n in range(nMax):
        an = (2/T)*np.trapz(f(xFuncPeriod)*np.cos(2*n*np.pi*xFuncPeriod*(1/T)), x=xFuncPeriod)
        bn = (2/T)*np.trapz(f(xFuncPeriod)*np.sin(2*n*np.pi*xFuncPeriod*(1/T)), x=xFuncPeriod)

        prev = np.add(an * np.cos(2*n*np.pi*x*(1/T)), bn * np.sin(2*n*np.pi*x*(1/T)))
        series = np.add(series, prev)

    return series, prev

def fourierMain(function, nMax, showPrevTerm):
    """
        Main function for calling produceFourierSeries and plotting

        ----------

        Parameters

        ----------

        function: string
            function key

        nMax: int
            number of terms
        
        showPrevTerm: bool
            true if most recent term should be displayed. False otherwise
    """

    # window upper and lower
    xMin = -np.pi
    xMax = np.pi

    x = np.arange(xMin, xMax, 0.001)

    functions = {'Linear':[f1, -np.pi, np.pi],'Square Wave':[f2, -np.pi/2, np.pi/2],'Cubic':[f3, -np.pi, np.pi]}
    f, x0, x1 = functions[function]

    _, ax = plt.subplots(figsize=(16,10))

    seriesApprox, prev = produceFourierSeries(x, nMax, x0, x1, f)

    ax.plot(x, seriesApprox, label='Fourier, {} terms'.format(nMax), color='C0')
    # plot prev term if necessary. Don't plot if no prev term. (Shouldnt be possible with slider lim anyway)
    if (nMax > 1) and showPrevTerm: ax.plot(x, prev, label='Latest Term', color='g')

    ax.plot(x, f(x), label='Actual', color='C1')

    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-4,4)
    ax.legend()
    plt.show()

def FourierSeries():
    """ 
        Main function called by notebook
    """
    terms_sldr = widgets.IntSlider(value=3, description='Terms', min=2, max=100, display=False, continuous_update=False)
    func_drp = widgets.Dropdown(options=['Linear','Square Wave','Cubic'],description='Function')
    prevTerm_check = widgets.Checkbox(value=False, description='Show most recent term')
    return widgets.VBox([widgets.HBox([terms_sldr, func_drp, prevTerm_check]), widgets.interactive_output(fourierMain, {'function':func_drp, 'nMax':terms_sldr, 'showPrevTerm':prevTerm_check})])

# 
# Signal Processing Code
# 

# some useful parameters

# min and max frequency for genwaves
minf = 1
maxf = 5

# file names and descriptor in dropdown
files = {'Guitar A string':'anote.wav', 'Bell C6':'c3Bell.wav', 'Synth':'generatedAudio.wav'}
path = "sound/"
# both these values depend on the exact .wav file you have. 
# Must be right for the frequency readout to be correct but will still give general shape
sampFreq = 44100. # sample frequency, MUST be a float
bitDepth = 16

def genWaves(noWaves, seed):
    """ produces amplitude and angular velocity of waves and places them in an array noWaves long with [a, w, s]
        where:

        ----------

        Parameters

        ----------

        noWaves: int
            number of waves to produce

        seed: int
            seed of random number generator

        ----------

        Returns

        ----------

        array_like
            noWaves by 3 with, a = amplitude, w = angular velocity s = horizontal shift

    """
    np.random.seed(seed)

    waves = []
    for _ in range(noWaves):
        f = np.random.uniform(minf,maxf)
        a = np.random.uniform()/(f) # crude downward bias on the amplitude of high frequencies
        s = np.random.uniform(0, 2*np.pi)

        waves.append([a, f, s])
    return waves

def plotWaves(waves):
    """ Main plotting function for RandomWave. Also sums waves to produce signal

        ----------

        Parameters

        ----------

        waves: array_like
            noWavesx3 array that contains amplitude, angular velocity and horizontal shift of each wave
    """
    _, (rawAx, compAx, ampAx) = plt.subplots(3,figsize=(16,10))
    x = np.arange(0,2*np.pi,0.01)
    out = 0
    for wave in waves:
        a, f, s = wave
        w = 2*np.pi*f
        out += a*np.sin(w*x-s)
        
        compAx.plot(x, a*np.sin(w*x-s), ls="--",linewidth=0.5)
    
    rawAx.plot(x, out, linewidth=2)
    freqAmp = np.array(waves).T
    order = np.argsort(freqAmp[1])
    freq = freqAmp[1][order]
    amp = freqAmp[0][order]
    ampAx.bar(freq, amp, width=0.1)
    
    compAx.set_title("Broken into Components")
    rawAx.set_title("Raw Signal")
    ampAx.set_xlim(minf,maxf)
    ampAx.set_xlabel("Frequency")
    ampAx.set_ylabel("Amplitude")
    ampAx.set_title("Frequency vs. Amplitude of Components")

    compAx.xaxis.set_ticklabels([])
    compAx.yaxis.set_ticklabels([]) # Removing all trace of axis labels because it looks prettier
    rawAx.xaxis.set_ticklabels([])
    rawAx.yaxis.set_ticklabels([])
    
    compAx.yaxis.set_ticks([])
    rawAx.xaxis.set_ticks([])
    compAx.xaxis.set_ticks([])
    rawAx.yaxis.set_ticks([])
    plt.show()

def runWaves(noWaves, seed, filterRange=[minf, maxf]):
    """ Filters waves from genWaves
        ----------

        Parameters

        ----------

        noWaves: int
            number of waves to produce

        seed: int
            seed for random number generator

        filterRange: array_like
            pair of min and max filter values
    """
    filterMin, filterMax = filterRange
    waves = genWaves(noWaves, seed)
    # filter frequency outside allowed range
    filteredWaves = [wave for wave in waves if((wave[1] < filterMin) or wave[1] > filterMax)] 
    if len(filteredWaves) > 1: plotWaves(filteredWaves)
    
def RandomWave():
    """ Main function called by notebook
    """
    noWaves_sldr = widgets.IntSlider(value=10, min=2, max=20, step=1, description='No. Waves', continuous_update=False)
    seed_Text = widgets.IntText(123, description='Seed')
    filter_sldr = widgets.FloatRangeSlider(value=[minf, minf], 
                                           min=minf, 
                                           max=maxf, 
                                           description="Filter Range", 
                                           continuous_update=False)
    return widgets.VBox([
        widgets.HBox([noWaves_sldr,filter_sldr,seed_Text]), 
        widgets.interactive_output(runWaves, {'noWaves':noWaves_sldr, 'seed':seed_Text, 'filterRange':filter_sldr})])

##########

# Fourier Analysis

##########

def readWav(fileName):
    """
        Reads data from .wav

        ----------

        Parameters

        ----------

        fileName: str
            name of file or path

        ----------

        Returns

        ----------

        Raw audio data

    """

    _, data = wavfile.read(path+fileName) # load the data

    audioData = data.T
    # check if wav is stereo
    if len(data.shape)>1: 
        audioData = audioData[0]
    return audioData

def processSignal(audioData):
    """
        Performs actual transform and returns:
        raw wav data
        transformed signal
        frequencies of transformed data

        ----------

        Parameters

        ----------

        audioData: array_like
            raw audio data

        ----------

        Returns

        ----------

        audioData: array_like
            Raw audio data

        tranformed: array_like
            frequency domain data

        freqs: array_like
            frequencies
        

    """
    tranformed = (rfft(audioData)) # perform transform

    freqs = rfftfreq(len(tranformed), d=1/sampFreq)

    return [audioData, tranformed, freqs]

def FourierAnalysis():
    """
        Called in notebook. Returns fft'd and raw data from bell and guitar .wavs.
        Calls the processSignal.

        for each file returns: [audio, tranformed, freqs]

    """
    # this is where I got the files from:
    #       guitar: https://freesound.org/people/casualdave/sounds/44729/ 
    #       bell  : https://freewavesamples.com/steel-bell-c6
    #       I made synth

    out = []

    for _, value in files.items():
        out.append(processSignal(readWav(value)))

    return out

def MusicNote(audioData, freqDom, freqs):
    """
        Produces UI componants for user. Drives plotting of PlotFourierAnalysis

        ----------

        Parameters

        ----------

        audioData: array_like
            raw audio data
        
        freqDom: array_like
            transformed frequency domain data
        
        freqs: array_like
            frequencies
    """
    xlim_sldr = widgets.IntRangeSlider(value=[0, 22.5e3], min=0, max=22.5e3, step=1000, continuous_update=False, description='Ax. lim')

    return widgets.VBox([widgets.interactive_output(PlotSignal, {
            'signal':widgets.fixed(audioData), 
            'amps':widgets.fixed(freqDom), 
            'freqs':widgets.fixed(freqs), 
            'FALim':xlim_sldr}), 
        widgets.HBox([xlim_sldr])])

def FilterFreqs(audioData, freqDom, freqs):
    """
        Produces UI componants for user. Drives plotting of filtered signal (FilterFreqs)

        ----------

        Parameters

        ----------

        audioData: array_like
            raw audio data
        
        freqDom: array_like
            transformed frequnecy domain data
        
        freqs: array_like
        frequencies
    """
    filter_sldr = widgets.IntRangeSlider(value=[0,0], min=0, max=23e3, step=1000, continuous_update=False, description='Freq Band')
    xlim_sldr = widgets.IntRangeSlider(value=[0, 22.5e3], min=0, max=22.5e3, step=1000, continuous_update=False, description='Ax. lim')
    export_btn = widgets.ToggleButton(value=False, description='Export to .wav')

    display(widgets.VBox([widgets.interactive_output(FilterBand, {
            'audioData':widgets.fixed(audioData), 
            'freqDom':widgets.fixed(freqDom), 
            'freqs':widgets.fixed(freqs), 
            'filtFreq':filter_sldr,
            'FALim':xlim_sldr,
            'export':export_btn
            }),
        widgets.HBox([xlim_sldr, filter_sldr, export_btn])]))
    
def FilterBand(audioData, freqDom, freqs, filtFreq, FALim, export):
    """
        Sets amplitudes in transform to 0 that are outside frequency range. Performs inverse tranform
        Calls PlotSignal

        ----------

        Parameters

        ----------

        audioData: array_like
            raw audio data

        processed: array_like
            fourier transformed data

        freqs: array_like
            frequencies for filtering and inversion

        filtFreq: array_like
            pair of frequencies representing filter band

        FALim: array_like
            pair of values representing x axis limits for analysis plot
    """
    # list comprehension alternative. Slower.
    # filteredTrans = [0 if (freqs[i] >= filtFreq[0]) and (freqs[i] <= filtFreq[1]) else freqDom[i] for i in range(len(freqDom))]

    # to store filtered transformed data
    # filteredTrans = np.zeros(len(freqDom))
    filteredTrans = freqDom.copy()
    for i in range(len(freqDom)):
        if ((freqs[i] >= filtFreq[0]) and (freqs[i] <= filtFreq[1])):
            filteredTrans[i] = 0

        if freqs[i] > filtFreq[1]: break

    # invert to filtered signal
    filteredSignal = irfft(filteredTrans)
    # slice out negatives
    filteredSignal = (filteredSignal[:int(len(filteredSignal)/2)])

    if export:
        exportWav(filteredSignal)

    PlotSignal(filteredSignal, filteredTrans, freqs, FALim)

def PlotSignal(signal, amps, freqs, FALim):
    """ 
        Inverts and plots filterd signal. 
        2 figures, freq and time domains.
        Called by FilterBand and MusicNote

        ----------

        Parameters

        ----------

        signal: array_like 
            audio data to be plotted on top plot ()

        amps: array_like
            amplitude data for fourier analysis graph

        freqs: array_like
            freqencies for analysis graph

        FALim: array_like
            Fourier analysis plot x limits
    """

    _, (ax1, ax2) = plt.subplots(2, figsize=(16,8))

    ax1.plot(signal, linewidth=0.1)
    ax1.set_title('Signal')
    ax1.set_xlim(0)
    ax1.set_ylim(-35e3, 35e3)

    # commenting scale on x because samples arent super useful. Can convert to seconds.
    ax1.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([])

    ax2.set_title('Fourier Analysis of Signal')
    ax2.plot(freqs, np.abs(amps), 'r', linewidth=0.5)
    ax2.set_yscale('log')
    ax2.set_xlim(*FALim)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_ylim(1e-3, 1e9)
    
    plt.show()

def exportWav(audioData):
    """ Function for exporting signal to wav

        ----------

        Parameters

        ----------

        audioData: array_like
            discrete audio signal
    """
    audioData = np.asarray(audioData, dtype=np.int16)
    
    wavfile.write(path+'exportedAudio.wav',int(sampFreq), audioData)


##########

# Even odd

##########

def oddSquare(x):
    ''' Square wave 
               ~ -1, -1 < x < 0
        f2(x)= |
               ~  1,  0 < x < 1
    '''
    return signal.square(2*x)

def evenSquare(x):
    ''' Square wave 
               ~ -1, -1 < x < 0
        f2(x)= |
               ~  1,  0 < x < 1
    '''
    return signal.square(2*(x + np.pi/4))

def cos2(x):
    return np.cos(2*x)

def sin2(x):
    return np.sin(2*x)

def quad(x):
    return x**2

def EvenOdd():
    funcs = {"cos(x)":np.cos, "cos(2x)":cos2,"cos(2x)":cos2,"sin(x)":np.sin, "x^2":quad, "Odd Square":oddSquare, "Even Square":evenSquare}
    f1_drop = widgets.Dropdown(options=funcs)
    f2_drop = widgets.Dropdown(options=funcs)
    sym_check = widgets.Checkbox(value=False, description="Check Symmetry")
    area_check = widgets.Checkbox(value=False, description="Integration Region")

    display(
        widgets.VBox([
            widgets.HBox([
                f1_drop,
                f2_drop,
                sym_check,
                area_check
                ]),
            widgets.interactive_output(symmetryCheck, {'f1':f1_drop, 'f2':f2_drop, "showSym":sym_check, "showArea":area_check})
        ])
    )

def symmetryCheck(f1, f2, showSym, showArea):
    
    a_slider = widgets.FloatSlider(value=1, min=0, max=np.pi, step=0.1, continuous_update=False)

    if not showSym:
        displayEvenOdd(f1, f2, showArea)
    else:
        display(
            widgets.VBox([
                a_slider,
                widgets.interactive_output(displayEvenOdd, {"f1":widgets.fixed(f1), "f2":widgets.fixed(f2), "showArea":widgets.fixed(showArea), "a":a_slider})
            ])
        )
    

def displayEvenOdd(f1, f2, showArea, a=None):
    
    x = np.linspace(-np.pi, np.pi,200)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 8))
    fig.tight_layout()

    ax1.plot(x, f1(x), color='C0')
    ax1.set_title('Function 1')
    ax2.plot(x, f2(x), color='C1')
    ax2.set_title('Function 2')
    ax3.plot(x, np.multiply(f1(x), f2(x)), color='C3')
    ax3.set_title('Function 1 * Function 2')

    if a is not None:
        ax1.plot(a, f1(a), marker=".", markersize="15", color="C4", label="f({})".format(a))
        ax1.plot(-1*a, f1(a), marker=".", markersize="15", color="C6", label="f(-{})".format(a))

        ax2.plot(a, f2(a), marker=".", markersize="15", color="C4", label="f({})".format(a))
        ax2.plot(-1*a, f2(a), marker=".", markersize="15", color="C6", label="f(-{})".format(a))

        ax1.legend()
        ax2.legend()

    if showArea:

        ax1.fill_between(x, f1(x), np.zeros(len(x)), color='C0', alpha=0.1)
        ax2.fill_between(x, f2(x), np.zeros(len(x)), color='C1', alpha=0.1)

    ax1.axvline(0, color='k', linewidth=0.5)
    ax2.axvline(0, color='k', linewidth=0.5)
    ax3.axvline(0, color='k', linewidth=0.5)

    ax1.axhline(0, color='k', linewidth=0.5)
    ax2.axhline(0, color='k', linewidth=0.5)
    ax3.axhline(0, color='k', linewidth=0.5)

    ax1.set_xlim(-np.pi, np.pi)
    ax2.set_xlim(-np.pi, np.pi)
    ax3.set_xlim(-np.pi, np.pi)