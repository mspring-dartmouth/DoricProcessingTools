__version__ = '2.6.1'


import pandas as pd
import numpy as np
import warnings
import seaborn as sb
from scipy import signal, optimize
import random
import pickle as pkl

from ._doric import h5read # A set of functions written and provided by Doric. 


###
# AUXILIARY FUNCTIONS
###

def butter_lowpass(cutoff, nyq_freq, order=4):
    if float(cutoff)/nyq_freq >= 1.0:
        warnings.warn(f'Target cutoff frequency ({cutoff}) for lowpass filter exceeds signal nyquist frequency ({nyq_freq}).\n\
                        Cutoff frequency will be set to nyquist frequency - 1 {nyq_freq-1}.')
        cutoff = nyq_freq - 1


    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def dbl_decay(params, y0, plat, x_vals):
    kfast, kslow, percent_fast = params
    span_fast = (y0-plat)*percent_fast
    span_slow = (y0-plat)*(1-percent_fast)
    Y = plat + span_fast*np.exp(-1*kfast*x_vals) + span_slow*np.exp(-1*kslow*x_vals)
    return Y
    
def fit_decay_double(params, y0, plat, true_y, x_vals):
    fitted_y = dbl_decay(params, y0, plat, x_vals)
    rss = np.sum((true_y-fitted_y)**2)
    return rss


def trap_sum(y, x):
    firstPointIDX = 0
    secondPointIDX = 1
    AUC = 0

    while secondPointIDX < len(y):
        rectArea = y[firstPointIDX] * abs(x[secondPointIDX]- x[firstPointIDX])
        triArea = 0.5 * abs(x[secondPointIDX]- x[firstPointIDX]) * abs(y[secondPointIDX]- y[firstPointIDX])
        AUC += rectArea + triArea
        firstPointIDX +=1
        secondPointIDX +=1

    return AUC

def bidirectional_trap_sum(y, x):
    '''
        Calculate the area under the curve with respect to 0, counting sections below zero as negative
        towards the final output. Accomplishes this by inserting 0s at all transition points, and calculating
        the AUC between each set of y=0 pairs. 

        param y: list-like containing y-values.
        param x: list-like containing x-values.
        return:  Area under the curve (as float) according the stated calculation above. 
    '''


    # Sanitize input type to array, and create copies.
    yArray = np.array(y)
    xArray = np.array(x, dtype='float64')

    # Identify points where signal crosses zero (np.diff(yArray>=0))
        # FALSE - FALSE = FALSE
        # TRUE - TRUE = FALSE
        # TRUE - FALSE = TRUE
        # FALSE - TRUE = TRUE
    
    # Use the indexer created by the above logic to get a list of indices at which signal
    # changes sign.
    inflectionPoints = np.where(np.diff(yArray>=0))[0]


    # Store inflection points as coordinate pairs on either side of 
    # each transition.
    transitionSections = {'x': [], 'y': []}
    inflectionsAtZeros = []
    for i in inflectionPoints:
        # If 0 is already a value in the signal, nothing need be done with this point.
        if 0 in y[i:i+2]:
            inflectionsAtZeros.append(i)
        # Otherwise, grab the line segment that crosses 0
        else:
            transitionSections['y'].append(yArray[i:i+2])
            transitionSections['x'].append(xArray[i:i+2])

    # Don't insert 0s where they already exist. Delete these indices from the iterable.

    indices_of_inflections_to_delete = []
    for inflection in inflectionsAtZeros:
        indices_of_inflections_to_delete.extend(np.where(inflectionPoints==inflection)[0])

    inflectionPoints = np.delete(inflectionPoints, indices_of_inflections_to_delete)

    # Calculate x values at which to y=0
    inflectionXs = []
    for i in range(len(inflectionPoints)):
        y1, y2 = transitionSections['y'][i]
        x1, x2 = transitionSections['x'][i]
        
        # Calculate the portion of the section between y1 and 0, and then
        # Multiply the total run by the same portion to determine how much to add
        # to x1 to get the target value. 
        target_x = x1 + (((0-y1)/(y2-y1))*(x2-x1))

        inflectionXs.append(target_x)

    
    # Insert 0s and corresponding x values into xArray and yArray.
    xArray = np.insert(xArray, inflectionPoints+1, np.array(inflectionXs))
    yArray = np.insert(yArray, inflectionPoints+1, np.zeros([1, inflectionPoints.size]))



    #  With 0s added at all transition points, identify these points.
    zeroBorders = np.where(yArray==0)[0].astype('int')

    # If the signal happens to start at 0, omit this point. 
    if np.any(zeroBorders==0):
        zeroBorders = zeroBorders[1:]

    # Pair up identified transition points to create sections for AUC calculation
    if len(zeroBorders)>0:
        signalWindows = [(0, zeroBorders[0])]
        signalWindows.extend([(start, stop) for start, stop in zip(zeroBorders[:-1], zeroBorders[1:])])
        signalWindows.append((zeroBorders[-1], yArray.size))       
    # ...unless the entirety of the signal is already all on one side of 0.
    else:
        signalWindows = [(0, yArray.size)]

    # Iterate over identified windows and calculate AUC for each one. 
    windowAUCs = []
    windowSigns = []
    for borders in signalWindows:
        signalSlice = yArray[borders[0]:borders[1]]
        xValSlice = xArray[borders[0]:borders[1]]

        # AUC is always positive because it is calculated on the absolute signal.
        windowAUCs.append(trap_sum(abs(signalSlice), xValSlice))

        # Determine whether the AUC should be considered positive or negative based on max value.
        if signalSlice.max() > 0:
            windowSigns.append(1)
        else:
            windowSigns.append(-1)


    # "Sign" and sum all AUCs to calculate final value.
    return sum(np.array(windowAUCs)*np.array(windowSigns))


def calc_robust_z(input_signal, ref_start_idx = 0, ref_end_idx = 'end'):
    '''
        Calculates a Robust Median Z scaled to the standard normal distribution relative to a specified window (none by default).
        param input_signal:               Raw input signal to transform (must be np.array() and may be either 1 or 2 dimensional)
        params ref_start_idx,ref_end_idx: The indices of the window on which to base the median/MAD calculation. Full signal by default. 
        return normalized_signal:         Median z-scaled signal
    '''

    if ref_end_idx == 'end':
        ref_end_idx = input_signal.size

    # If 1 dimensional array, calculation is simple.
    if len(input_signal.shape) == 1:
        med = np.nanmedian(input_signal[ref_start_idx:ref_end_idx])
        MAD = np.nanmedian(abs(input_signal[ref_start_idx:ref_end_idx]-med))*1.4826

    # If 2 dimensional array, perform the calculation simultaneously for each row independently. 
    elif len(input_signal.shape) == 2:
        med = np.nanmedian(input_signal[:, ref_start_idx:ref_end_idx], axis=1).reshape([-1, 1])
        MAD = np.nanmedian(abs(input_signal[:, ref_start_idx:ref_end_idx] - med))*1.4826

    normalized_signal = (input_signal - med) / MAD
    return normalized_signal


class RenamingUnpickler(pkl.Unpickler):
    '''
        Pickles created prior to the implementation of a pip installable version of this module
        were created under a different name. In changing the name, the old pickles become unreadable
        by a system expecting to find doricprocessingtoolbox instead of Doric_Processing_Toolbox.
        This is an adaptation of the base pkl.Unpickler class in which the find_class function has been
        modified to fix the issue above.

        If pd.read_pickle(pkl) throws a ModuleNoteFound error when trying to load an older pickle, 
        attempt to load it using RenamingUnpickler(pkl).load() instead. 
    '''
    def find_class(self, module, name):
        if module == 'Doric_Processing_Toolbox':
            module = 'doricprocessingtoolbox'
        return super().find_class(module, name)



# SIGNAL PROCESSING OBJECT CLASS

# TODO: Comment __init__ function and class more generally. 
class sig_processing_object(object):
    def __init__(self, input_file, remove_artifacts=True, from_pickle=False):
        '''

        '''


        # INITIATE FROM RAW DATA
       
        if not from_pickle:
            # These two are used in the same way regardless of input format. 
            self.signal_processing_log = []
            self.trial_data = {} # This will be used in align_to_TTLs to store TTL aligned data.

            if '.csv' in input_file:
                self.input_data_frame = pd.read_csv(input_file, skiprows=1, error_bad_lines=False, warn_bad_lines=True)
                
                # Purge extraneous columns
                # AOut-1/2 are just the sin functions corresponding to the LED driver.
                for col in ['AOut-1', 'AOut-2']:
                    if col in self.input_data_frame.columns:
                        self.input_data_frame.drop(col, axis=1, inplace=True)
                # Occasionally, empty columns will be picked up. They are named "Unnamed: X", where X is the column number
                unnamed_col_locs = ['Unnamed' in i for i in self.input_data_frame.columns]
                unnamed_col_ids = self.input_data_frame.columns[unnamed_col_locs]
                self.input_data_frame.drop(unnamed_col_ids, axis=1, inplace=True)

                # Now rename columns to something more recognizable.
                input_file_col_names = {'Time(s)': 'Time', 
                                        'AIn-1 - Dem (AOut-1)': 'Signal',
                                        'AIn-1 - Dem (AOut-2)': 'Isosbestic',
                                        'AIn-1 - Raw': 'TotalRaw',
                                        'DI/O-1': 'TTL_1',
                                        'DI/O-2': 'TTL_2',
                                        'DI/O-3': 'TTL_3',
                                        'DI/O-4': 'TTL_4'}
                try:
                    new_col_names = [input_file_col_names[n] for n in self.input_data_frame.columns]
                except KeyError:
                    print('Not all column names were found in renaming dictionary.')
                    new_col_names = []
                    # If the columns can't be renamed in one go, then you'll just have to do it iteratively.
                    for old_col_name in self.input_data_frame.columns:
                        # If the column name is one of the ones you expected, change it to the standard.
                        if old_col_name in input_file_col_names.keys():
                            new_col_names.append(input_file_col_names[old_col_name])
                        # Otherwise, just leave it untouched, and let the user know.
                        else:
                            print(f'{old_col_name} not found in input_file_col_names.')
                            new_col_names.append(old_col_name)

                for old_name, new_name in zip(self.input_data_frame.columns, new_col_names):
                    print(f'{old_name}-->{new_name}')   
                self.input_data_frame.columns = new_col_names
                # Remove any rows with missing data.
                self.input_data_frame.dropna(how='any', inplace=True)

                # Set up attributes.
                self.sampling_rate = 1/np.diff(self.input_data_frame.Time).mean()
                self.timestamps = self.input_data_frame.Time.values
                self.signal = self.input_data_frame.Signal.values
                self.isosbestic = self.input_data_frame.Isosbestic.values


            elif '.doric' in input_file:
                # If the raw data are in an HDF5 binary file, a different set of processing steps is required. 

                self.signal, exc_info = h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'AIN01xAOUT01-LockIn', 'Values'])
                exc_time, exc_time_info = h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'AIN01xAOUT01-LockIn', 'Time'])


                self.isosbestic, iso_info = h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'AIN01xAOUT02-LockIn', 'Values'])
                iso_time, iso_time_info = h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'AIN01xAOUT02-LockIn', 'Time'])

                if (abs(exc_time - iso_time)<0.05).all():
                    self.timestamps = exc_time
                    self.sampling_rate = 1/np.diff(self.timestamps).mean()
                else:
                    raise Exception('Excitation and Isosbestic Timestamps not aligned. Check your data.')
                


                # identify_TTLs will look for an attribute named self.input_data_frame and select only the TTLs. 
                # Put the TTL data in that format for simplicity's sake. 
                
                ttl_time, ttl_time_info = h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'DigitalIO', 'Time'])

                self.input_data_frame = pd.DataFrame(index=range(ttl_time.size), columns=['Time', 'TTL_01', 'TTL_02', 'TTL_03', 'TTL_04'])

                self.input_data_frame.loc[:, 'Time'] = ttl_time.copy()

                for ttl_num in range(1, 5):
                    try:
                        raw_data, inf =  h5read(input_file, ['DataAcquisition', 'FPConsole', 'Signals', 'Series0001', 'DigitalIO', f'DIO0{ttl_num}'])
                        self.input_data_frame.loc[:, f'TTL_0{ttl_num}'] = raw_data.copy()
                    except KeyError:
                        print(f'{input_file} has no TTL {ttl_num}.')
                        break

                # Now get rid of any empty columns
                self.input_data_frame.dropna(axis=1, inplace=True)
                
                # Delete a few things that are duplicated to preserve memory
                del ttl_time
                del raw_data


            # Delete the input data frame to avoid memory issues
            # Capitalize on greater temporal specificity of raw input for calculating TTL times before doing so. 
            self.identify_TTLs()
            # If there are no TTLs, this will simply create an empty dictionary named self.ttl_starts.
            del self.input_data_frame

            # The doric system introduces a strange artifact every ~937 seconds in which the signal on both channels cuts out. 
            # Remove these artifacts. 
            
            artifact_buffer = int(4 * self.sampling_rate / 1000) # Yields number of frames in ~4 ms. 
            if remove_artifacts:
                self.remove_artifacts(reference_channel='Isosbestic', threshold=-10, buffer_size = artifact_buffer)
                self.remove_artifacts(reference_channel='Signal', threshold=-15, buffer_size = artifact_buffer)
                # Occasionally, I will note that a clear artifact occurs only in the signal channel. As such, filtering should occur using
                # both channels as the references, but we will be more stringent using the Signal channel as the reference. 

            # Apply lowpass butterworth filter to isosbestic channel.   
            self.isosbestic = butter_lowpass_filter(self.isosbestic, 40, self.sampling_rate/2, order=4)
            self.signal_processing_log.append(f'Butterworth lowpass filter (40Hz) applied to isosbestic channel.')
            # By performing this step here, you can apply whatever downsample you want to the processed data, because 
            # the isosbestic will not have to be refiltered.  
        

        # INITIATE FROM PICKLED SPO
        if from_pickle:
            # Load the file:
            with open(input_file, 'rb') as in_file:
                try:
                    input_spo = pd.read_pickle(in_file)
                except ModuleNotFoundError:
                    in_file.seek(0)
                    input_spo = RenamingUnpickler(in_file).load()
            # VERSION CHECKING
            try: 
                major_version, minor_version, patch_version = input_spo.__version__.split('.')
            except AttributeError:
                # If the spo was created before version control was implemented, it is from v1.1.0
                major_version, minor_version, patch_version = ('1', '0', '0')
                input_spo.__version__ = '1.0.0' # This is just to make referring to this easier down the line. 

            if major_version != __version__.split('.')[0]:
                # Force exit if there's likely to be issues with backwards compatability. 
                raise Exception(f'Version of input_file {input_spo.__version__} is not supported. \
                                  Current Version of Doric Processing Toolbox is {__version__}.')

            elif input_spo.__version__ != __version__:
                # If there aren't likely to be any issues, at least let your user know if there's a difference. 
                warnings.warn(f'Reading in an old signal_processing_object: v{input_spo.__version__}. Current version is v{__version__}.')
            
            # ACTUAL IMPORT

            # The easiest way to do this is just to iterate over the existing attributes within the old spo and store them in the new one. 
            for attribute_name, attribute_value in input_spo.__dict__.items():
                self.__dict__[attribute_name] = attribute_value

        # Track the version number.  
        self.__version__ = __version__

    def downsample_signal(self, downsample_factor):
        
        #Prior to downsampling, determine target sampling rate and apply anti-aliasing filter.
        target_sampling_rate = self.sampling_rate / downsample_factor
        new_nyq = target_sampling_rate/2

        # Apply filter to signal and isosbestic to remove all frequencies above the nyquist frequency of the downsampled sampling rate.
        self.signal = butter_lowpass_filter(self.signal, new_nyq, self.sampling_rate/2)
        self.isosbestic = butter_lowpass_filter(self.isosbestic, new_nyq, self.sampling_rate/2)
        
        self.signal_processing_log.append(f'{new_nyq}Hz anti-aliasing filter applied to Signal and Isosbestic.')
        
        downsampler_index = np.arange(0, self.signal.size, downsample_factor)

        self.timestamps = self.timestamps[downsampler_index]
        self.signal = self.signal[downsampler_index]
        self.isosbestic = self.isosbestic[downsampler_index]


        if 'deltaF/F calculated.' in self.signal_processing_log:
            warnings.warn(f'Input data downsampled. Re-calculate DFF.')
            try:
                del self.processed_dataframe
                self.signal_processing_log.append('self.processed_dataframe removed following downsampling.')
            except AttributeError:
                pass



        self.signal_processing_log.append(f'Signal downsampled by a factor of {downsample_factor}. {self.sampling_rate}-->{target_sampling_rate}')
        self.sampling_rate = target_sampling_rate
        return self.signal_processing_log


    def filter_signals(self, filtering_threshold=10, order=4):
        '''
			Applies a 4th order 10Hz (by default) low-pass butterworth filter to the isosbestic and excitation channels. 
			:param self: current attributes of the signal_processing_object
			:param filtering_threshold: the maximum Hz to permit in the signal.
			:param order: the filter order to use. Higher orders sacrifice accuracy for steeper drop-offs after the cut-off. 4 seems a good balance. 
        ''' 
        self.isosbestic = butter_lowpass_filter(self.isosbestic, filtering_threshold, self.sampling_rate/2, order=order)
        self.signal = butter_lowpass_filter(self.signal, filtering_threshold, self.sampling_rate/2, order=order)
        self.signal_processing_log.append(f'{order}th order Butterworth lowpass filter ({filtering_threshold}Hz) applied to isosbestic and signal channels.')


        if hasattr(self, 'processed_dataframe'):
            warnings.warn('Raw traces have been filtered. Detrending, movement correction, and normalization must be re-executed.')


        return self.signal_processing_log

    # Fit the processed isosbestic signal was fitted to the excitation signal using a linear fit to correct for signal decay. 
    def calc_dff_from_isosbestic(self):
        '''
            A function for calculating the DeltaF/F based on the isosbestic. See Lerner et al. (2015) Cell. 10.1016/j.cell.2015.07.014. 

            This function filters the isosbestic channel, fits it to the signal channel, and then calculates dF/F from the fit.
                param   self:                       current attributes of the signal_processing_object
                create  self.fitted_isosbestic:     array containing filtered isosbestic with linear fit applied
                create  self.dff:                   array containing excitation signal normalized relative to the filtered and fitted isosbestic 
                return  self.signal_processing_log: list containing a record of processing steps so far applied to the data.
        '''

        # Fit isosbestic channel to signal channel using linear fit.
        fit_coefs = np.polyfit(self.isosbestic, self.signal, 1)
        self.fitted_isobestic = (fit_coefs[0] * self.isosbestic) + fit_coefs[1]

        self.signal_processing_log.append('Linear fit applied to isosbestic channel.')

        # Calculate the DeltaF / F. 
        self.dff = (self.signal - self.fitted_isobestic) / self.fitted_isobestic
        self.signal_processing_log.append(r'deltaF/F calculated based on isosbestic.')

        return self.signal_processing_log

     
    def calc_dff_from_percentile(self, seconds=10, ref_percentile=5):
        '''
            A function for calcuating the DeltaF/F using an alternate, isosbestic free method. 
            This method is based on Krok et al. in BioRxiv https://doi.org/10.1038/s41586-023-05995-9

            This function calculates the nth-percentile of the signal in a 10 second sliding window, 
            then uses this as F0 in the formula dFF = (F-F0) / F0, where F=465 signal. 
               param   self:                       current attributes of the signal_processing_object
               param   seconds:                    window size, in seconds, in which to perform the dFF calculation
               param   ref_percentile:             the percentile to use as the reference for the dFF calculation
               create  self.dff:                   array containing excitation signal normalized relative to the filtered and fitted isosbestic 
               return  self.signal_processing_log: list containing a record of processing steps so far applied to the data.
        '''

        ## Calculates the nth percentile of X-second bins in the 465 signal. 
        
        # The length of the signal will not be perfectly divisible by the number of seconds chosen, so need to calculate where the overflow is. 
        bin_size = int(self.sampling_rate * seconds)
        last_bin = (self.signal.size // bin_size) * bin_size
        
        # Reshape signal so that it is n rows of X second columns:
        main_sig_array = self.signal[:last_bin].reshape([-1, bin_size])
        remainder_sig = self.signal[last_bin:]

        # Then calculate the nth percentile for X second increments + the remainder. 
        # Main
        sig_percentile = np.percentile(main_sig_array, ref_percentile, axis=1).reshape([-1, 1])
        # Remainder
        try:
            sig_percentile_remainder = np.percentile(remainder_sig, ref_percentile)
        except IndexError:
            print(f'Remarkably, the signal length ({self.signal.size}) is perfectly divisible by the bin size ({bin_size}).')
            sig_percentile_remainder = np.array([])

        #Calculate dFF for both main and remainder:
        main_dff = (main_sig_array - sig_percentile) / sig_percentile
        remainder_dff = (remainder_sig - sig_percentile_remainder) / sig_percentile_remainder

        # Combine body and tail. 
        self.dff = np.append(main_dff.reshape([1, -1]), remainder_dff.reshape([1, -1]))

        self.signal_processing_log.append(f'deltaF/F calculated based on {ref_percentile}th percentile in {seconds}s windows.')


        return self.signal_processing_log

    def detrend_photobleaching(self):

        # Fit two-phsae decay curve to signal
        intercept, plateau = np.median(self.signal[:int(self.sampling_rate)+1]), np.min(self.signal)
        sig_fit = optimize.minimize(fit_decay_double, x0=[0.1, 0.001, 0.2], args=(intercept, plateau, self.signal, self.timestamps), method='Nelder-Mead')
        
        # Fit two-phsae decay curve to isosbestic
        iso_intercept, iso_plateau = np.median(self.isosbestic[:int(self.sampling_rate)+1]), np.min(self.isosbestic)
        iso_fit = optimize.minimize(fit_decay_double, x0=sig_fit['x'], args=(iso_intercept, iso_plateau, self.isosbestic, self.timestamps), method='Nelder-Mead')
             
        # Subtract decay curves from both signals. 
        self.detrended_sig = self.signal-dbl_decay(sig_fit['x'], intercept, plateau, self.timestamps)
        self.detrended_iso = self.isosbestic-dbl_decay(iso_fit['x'], iso_intercept, iso_plateau, self.timestamps)
      
        self.signal_processing_log.append('Signal and isosbestic detrended according to individual two-phase decay curves.')

        return self.signal_processing_log

    def correct_movement(self):
        try:
            coefs = np.polyfit(self.detrended_iso, self.detrended_sig, 1)
        except AttributeError as e:
            raise Exception('detrend_photobleaching() must be run prior to movement correction.') from e

        # Estimate motion over course of session based on isosbestic. 
        est_motion = coefs[1] + coefs[0] * self.detrended_iso

        # Correct GRAB signal by subtracting estimated motion from detrended signal. 
        self.motion_corrected_signal = self.detrended_sig - est_motion

        self.signal_processing_log.append('Motion component of signal estimated based on linear fit and removed from detrended signal.')

        # For compatability with existing functions:
        self.dff = self.motion_corrected_signal
        # Obviously, the motion corrected signal is NOT in units of DeltaF/F, and this hack will be repaired in future versions. 
        # For now, I just want to be able to implement the detrend + motion correct with existing save files and existing functions. 

        return self.signal_processing_log

    def z_norm_deltaff(self, normalization_window_size = 'None'):
        '''
            Convert DeltaF/F to Robust Z scores based on a sliding window. 
            param  self:                       attributes of signal_processing_object
            param normalization_window_size    The size of the sliding window (in seconds) to use for calculating normalized_signal.
            create self.normalized_signal:     robust z normalized DeltaF/F
            return self.signal_processing_log: list containing a record of processing steps so far applied to the data.
        '''
        if normalization_window_size == 'None':
            self.normalized_signal = calc_robust_z(self.dff)
            self.signal_processing_log.append(f'Robust Z-Score normalization performed on deltaF/F using global median and MAD.')
            return self.signal_processing_log
        
        # Convert window size into a chunk of indices using the internal sampling rate (samples/second).
        average_window_step_size = int(normalization_window_size*self.sampling_rate)
        
        # Confirm that self.dff has been created and generate the end cap for the while loop.
        try:
            signal_size = self.dff.size
        except AttributeError as e:
            raise Exception('Cannot normalize deltaF/F if deltaF/F has not been created! Run calc_dff().') from e
        
        # Initialize loop
        self.normalized_signal = np.array([])
        window_start = 0
        while window_start < signal_size:
            # Set bounds for calculation and check whether we're at the end
            window_end = window_start + average_window_step_size
            if window_end > signal_size:
                window_end = signal_size

            # Perform the normalization on the current window and store it
            self.normalized_signal = np.append(self.normalized_signal,
                                               calc_robust_z(self.dff[window_start:window_end]))
            
            # Increment start idx to avoid infinite loop.
            window_start = window_end
        

        self.signal_processing_log.append(f'Robust Z-Score normalization performed on deltaF/F using sliding {normalization_window_size}s window.')
        return self.signal_processing_log


    def remove_artifacts(self, reference_channel = 'Isosbestic', threshold = -10, buffer_size = 50):
        '''
            The artifacts present in data from the Doric system are strange and appear to be the product of the signal cutting out
            approximately once every 937 seconds. The best way to filter out crud is to operate directly on the signal and isosbestic.

            param self:                        attributes of sig_processing_object
            param reference_channel:           The channel (isosbestic or signal) to identify outliers in. 
            param threshold:                   The Z score threshold used to identify drops in signal
            buffer_size:                       The number of frames on either side of the identified artifact window to include in the smooth.
            return self.signal_processing_log: list containing a record of processing steps so far applied to the data.
        '''
        
        # Normalize the isosbestic signal using robust median z method
        if reference_channel == 'Isosbestic':
            sig_channel = self.isosbestic
        elif reference_channel == 'Signal':
            sig_channel = self.signal
        else:
            print(f'Method remove_artifacts does not recognize reference_channel "{reference_channel}". Channel must be Signal or Isosbestic. Defaulting to Isosbestic.')
            sig_channel = self.isosbestic

    
        med = np.median(sig_channel)
        mad = np.median(abs(sig_channel-med))*1.4826
        robust_z_trace = (sig_channel-med)/mad

        # Identify points at which robust_z_trace crosses threshold. 
        switch_points = np.diff(robust_z_trace<threshold, prepend=0)
        # Switch point has 3 possible values: 
        #   1: indicates that signal dropped below threshold (True - False)
        #  -1: indicates that signal came above threshold (False - True)
        #   0: Indicates that signal remained on one side of threshold (True - True | False - False)

        # Identify the points at which an artifact begins and ends. Step buffer_size frames to either side of the 
        #   threshold defined artifact window to ensure that it is entirely excised.
        art_start = np.where(switch_points==1)[0]-buffer_size
        art_end = np.where(switch_points==-1)[0]+buffer_size
        # N.B. The size of this step relative to the signal changes depending on the sampling rate of the file. 
        # The default buffer_size of 50 frames was set based on the default sampling rate of the Doric systems (12 kSpS)
        # to create a time buffer of approximately 4.1667 ms. 

        # Check that the last end-point + the buffer isn't past the end of the existing timestamps
        if art_end.size > 0 and art_end[-1] >= self.timestamps.size:
            art_end[-1] = self.timestamps.size-1


        # Directly remove artifacts from signal and isosbestic
        for start, end in zip(art_start, art_end):
            
            # The artifact period will be overwritten with values interpolated between two values on either side
            #   of the artifact. 

            # The first step in this process is to identify values on either side of the artifact: 

            # Time is common between signal and isosbestic:
            time_firstpoint, time_endpoint = (self.timestamps[start], self.timestamps[end])

            # Identify signal level independently for excitation signal and isosbestic:
            sig_fp, sig_ep = (self.signal[start], self.signal[end])
            iso_fp, iso_ep = (self.isosbestic[start], self.isosbestic[end])

            # Calculate the interpolated values ...
            timepoints_for_interp = self.timestamps[start:end]

            sig_interp = np.interp(x = timepoints_for_interp, 
                                   xp = [time_firstpoint, time_endpoint], 
                                   fp = [sig_fp, sig_ep])
            
            iso_interp = np.interp(x = timepoints_for_interp, 
                                   xp = [time_firstpoint, time_endpoint], 
                                   fp = [iso_fp, iso_ep])

            
            # and insert them into each
            self.signal[start:end] = sig_interp
            self.isosbestic[start:end] = iso_interp


        # Record the process
        self.signal_processing_log.append(f'Artifacts removed from Signal and Isosbestic using threshold={threshold} on {reference_channel}.')
        return self.signal_processing_log

    def create_dataframe(self):
        '''
            Combine current timestamps, DeltaF/F, and normalized DeltaF/F into DataFrame.
            param  self:                       attributes of signal_processing_object
            create self.processed_dataframe:   DataFrame with index = Timestamps, and Columns = raw and normalized DeltaF/F signal
            return self.signal_processing_log: list containing a record of processing steps so far applied to the data.
        '''
        try:
            self.processed_dataframe = pd.DataFrame(index=self.timestamps, columns = ['RawSignal', 'Z_Signal'], 
                                                    data=np.hstack([self.dff.reshape(-1, 1), self.normalized_signal.reshape(-1, 1)]))
            self.signal_processing_log.append('Timestamps, Raw Signal, and Normalized Signal combined into DataFrame (self.processed_dataframe).')
            return self.signal_processing_log
        except AttributeError as e:
            raise Exception('Missing attributes. Run calc_dff() and z_norm_deltaff().') from e

    def trim_signal(self, start_time, end_time):
        '''
            Trims self.signal, self.isosbestic, and self.timestamps.
            param self:                       attributes of signal_processing_object
            param start_time:                 minimum start time to keep after trimming, inclusive
            param stop_time:                  maximum start time to keep after trimming, inclusive          
            return self.signal_processing_log: list containing a record of processing steps so far applied to the data.

            usage: self.trim_signal(50, 100) to trim the signals from 50 to 100 seconds
                   self.trim_signal(100, 'end') to trim signals from 100 seconds to the end.
        '''
        if end_time == 'end':
            end_time = self.timestamps.max()

        trimmed_indices = np.where((self.timestamps>=start_time)&(self.timestamps<=end_time))
        self.timestamps = self.timestamps[trimmed_indices]
        self.signal = self.signal[trimmed_indices]
        self.isosbestic = self.isosbestic[trimmed_indices]

        self.signal_processing_log.append(f'Signal cropped to {start_time}s:{end_time}s.')
        
        if 'deltaF/F calculated.' in self.signal_processing_log:
            warnings.warn('Raw signal has been trimmed. Re-run calc_dff() and z_norm_delta() to perform normalizations on trimmed data.')

            try:
                del self.processed_dataframe
                self.signal_processing_log.append('self.processed_dataframe removed following signal trimming.')
            except AttributeError:
                pass

        return self.signal_processing_log


    def identify_peaks_GunaydinMethod(self):
        '''
            A heuristic for identifying peaks in signal. Taken from https://dx.doi.org/10.1016%2Fj.cell.2014.05.017
            param  self:                       attributes of signal_processing_object
            create self.peak_timestamps:       timestamps at which a peak has been identified
            return self.peak_timestamps:       ''

        '''
        warnings.warn('This method does not work well. Use id_transients instead. This method will be removed in 2.0.0')
        #1) take the derivative of the squared difference between two different low pass filtered copies of the time series data (at 40Hz and 0.4Hz);
        
        # Generate low pass filtered copies at 40Hz and 0.4Hz. 
        try:
            filt_40hz = butter_lowpass_filter(self.dff, 40, self.sampling_rate/2)
            filt_point4hz = butter_lowpass_filter(self.dff, 0.4, self.sampling_rate/2)
        except ValueError as e:
            raise Exception(f'Data must have sampling rate of >80Hz to apply Gunyadin method. Signal has sampling rate {self.sampling_rate}.') from e

        # Calculate the squared difference between the copies. 
        sq_diff = (filt_40hz - filt_point4hz)**2
        

        #2) threshold these values at 95% confidence interval (Z=2)
        
        # Normalize the signal for comparing to confidence interval threshold
        sq_diff_med = np.median(sq_diff)
        sq_diff_MAD = np.median(abs(sq_diff-sq_diff_med))*1.4826
        sq_diff_normalized = (sq_diff-sq_diff_med)/sq_diff_MAD

        # Calculate the derivative
        dy = np.diff(sq_diff_normalized, prepend=0)
        dx = 1/self.sampling_rate
        deriv = dy/dx

        # Identify points where derivative exceeds threshold
        deriv_peaks = np.where(deriv>=2)



        #3) classify these smooth derivative-based time points as ‘‘peaks’’ if absolute fluorescence measurement is above 95% confidence interval
        robust_z_peaks = np.where(abs(self.normalized_signal)>=2)
        self.peak_timestamps = np.intersect1d(self.timestamps[deriv_peaks], self.timestamps[robust_z_peaks])
        

        self.signal_processing_log.append('Fluorescent peaks (and/or troughs) identified according to Gunyadin et al., 2014.')


    def id_transients(self, min_duration = 0.5, debug=False):
        '''
            Peak detection algorithm according to https://doi.org/10.1126/science.aat8078
            param  self:                                attributes of signal_processing_object
            create self.transient_event_timestamps:     A dictionary containing key, value pairs of event_number, timestamps comprising the transient

        '''

        # Calculate normalized derivative of lowpass filtered signal with 5th order smooth (we're looking for gestalts here)
        dx = 1/self.sampling_rate
        lowpass_sig = butter_lowpass_filter(self.normalized_signal, 1, self.sampling_rate/2, order=4)
        deriv = np.diff(lowpass_sig, prepend=0)/dx
        normed_deriv = calc_robust_z(deriv)

        # Identify indices at which signal crosses threshold of 0.5 MADs in either direction
        sig_switch_points = np.diff(self.normalized_signal > 0.5, prepend=0)
        trans_starts, = np.where(sig_switch_points==1) # Transient starts intially identified by point where signal goes from <0.5 to >0.5.
        trans_ends, = np.where(sig_switch_points==-1)


        # What goes up, must come down. And if it doesn't come down, then it's because the end of the file cut it off.
        if trans_starts.size != trans_ends.size:
            trans_ends = np.append(trans_ends, sig_switch_points.size-1)
        
        # Determine the true transient start based on when the slope first becomes > 1.0 en route to signal crossing 0.5 MADs
        event_timestamps = {}

        # Some events identified using the simple threshold method will be the result of drift (i.e. fail derivative requirement)
            # or are perhaps better characterized as extensions of a previous transient, but the signal dropped below 0.5 MADs for a single frame. 
        events_to_delete = []
        events_to_join = []

        # Array for storing true transient starts based on derivative criteria
        slope_starts = np.array([], dtype='int')
        

        for event_number, event_start in enumerate(trans_starts):

            # Candidate start points are all indices prior to the already identified threshold-crossing
            # where the derivative is BELOW 1.0 MAD above the median slope
            candidates, = np.where(normed_deriv[:event_start+1]<1.0)
            # The start of the rise is the last such point before the threshold crossing. 
            # This allows the rise itself is classified as part of the transient.
            try:
                slope_starts = np.append(slope_starts, int(candidates.max()))
            except ValueError:
                # In the case that the recording itself begins with a spike in signal, there may be no valid candidates. 
                # Attempting to take the max() of an empty array returns a ValueError. 
                # If this is the case, then the slope can be said to start at bin 0 (the first frame of the recording)
                # This is of course only possible if we are looking at the very first event.
                if event_number == 0:
                    slope_starts = np.append(slope_starts, 0)
                    candidates = np.append(candidates, 0) # If you don't do this, you'll throw an error in the validity check below
                else:
                    raise ValueError # If the problem is something else, (even though I can't think of what it would be right now)
                                     # I don't want to let it pass. This is basically just here to catch any future bugs. 

            # Consider the possibilities identified prior to the regarding invalid transients
            if candidates.max() == event_start:
                # If the threshold crossing itself occurs with a low slope, it's either part of the previous transient ...
                if event_start - trans_ends[event_number-1] == 1:
                    # ... if and only if the current transient starts one frame after the previous ended. 
                    events_to_join.append((event_number-1, event_number))
                elif event_number == 0:
                    # However, if it is the first event, it can't be on the tail of the previous. See above, and let it be.
                    pass
                else: # If it isn't immediately on the tail of a previous event (or isn't itself the first event) scrap it. It's drift.
                    events_to_delete.append(event_number)
              
        # Linking events is as simple as extending the end point of the true transient
        # to become the end point of the false transient
        # It's best to work backwards to ensure that if there are multiple false transients 
        # (consider a case where the transient is just above threshold and is a little noisy)
        # The "true" end-point is carried backwards through the process.        
        events_to_join.reverse() # To this end, reverse the tuples of events.

        for main_event, tail_event in events_to_join:
            trans_ends[main_event] = trans_ends[tail_event]
            events_to_delete.append(tail_event)
        
        # Join events that are separated by a gap that is shorter than the minimum transient time
        transient_gaps = self.timestamps[slope_starts[1:]] - self.timestamps[trans_ends[:-1]]
        gaps_to_join, = np.where(transient_gaps<=min_duration)
        for idx in gaps_to_join[::-1]:
            trans_ends[idx] = trans_ends[idx+1]
            events_to_delete.append(idx+1)
        events_to_delete = list(set(events_to_delete))

        # trans_starts is omitted because it isn't used again. Cleaning it up doesn't matter.
        trans_ends = np.delete(trans_ends, events_to_delete)
        slope_starts = np.delete(slope_starts, events_to_delete)
        
        # Now record all the transient start, stop pairs AFTER filtering each for duration
        self.transient_event_timestamps = {}
        event_number = 0
        for event_start, event_end in zip(slope_starts, trans_ends):
            if self.timestamps[event_end] - self.timestamps[event_start] >= min_duration:
                self.transient_event_timestamps[event_number] = self.timestamps[event_start:event_end+1]
                event_number+=1
        
        self.signal_processing_log.append(f'Transients identified according to Liston et al., 2020. Min duration = {min_duration}s.')

        if debug:
            raise RuntimeError

    # TODO Comment these.
    def identify_TTLs(self):
        ttl_channels = filter(lambda x: 'TTL_' in x, self.input_data_frame.columns)

        self.ttl_starts = {}
        for ttl_ch in ttl_channels:

            switch_points = np.diff(self.input_data_frame.loc[:, ttl_ch], prepend=0)

            ttl_starts, = np.where(switch_points==1)
            
            if ttl_starts.size > 0:
                # Brief corner-case check:
                if (ttl_starts[0] == 0) and (self.input_data_frame.index[0] !=0):
                    ttl_starts[0] = self.input_data_frame.index[0]
                    # In the event that data have been dropped from the beginning of the file and 
                    # a TTL begins in the first frame of the resulting dataframe, the index of the first TTL
                    # will be recorded, incorrectly, as 0. It should be the first index of input_data_frame.

            self.ttl_starts[ttl_ch] = self.input_data_frame.loc[ttl_starts, 'Time'].values

        if len(self.ttl_starts) == 0:
            log_txt = 'Attempted to identify TTL onsets, but none were found.' 
        else:
            log_txt = 'TTL onset timestamps identified.'
        
        self.signal_processing_log.append(log_txt)


    def align_to_TTLs(self, reference_TTL = 'TTL_01', baseline_time = 10, epoch_time = 10, signal_to_slice='Z_Signal'):
        # Determine number of bins to devote to baseline.
        n_baseline_bins = int(baseline_time * self.sampling_rate)
        # The next frame after n_baseline_bins is where we want to ensure that our 0 ends up in all slices. 
        target_zero_index = n_baseline_bins+1

        slices = []
        for time in self.ttl_starts[reference_TTL]:
            # Take an initial slice of timestamps based on baseline and epoch lengths relative to ttl_start 
            timestamps_bools = (self.processed_dataframe.index>(time-baseline_time)) & (self.processed_dataframe.index<(time+epoch_time))
            timestamps = self.processed_dataframe.index[timestamps_bools]


            isi = 1 / self.sampling_rate
            real_sample_spacing = np.diff(timestamps)
            real_sample_spacing = np.insert(real_sample_spacing, 0, isi)

            off_indices = np.where(real_sample_spacing>=isi*1.05)[0]

            padded_signal = self.processed_dataframe.loc[timestamps, signal_to_slice].values

            for diff_idx in np.flip(off_indices):
                n_samples_skipped = np.round(real_sample_spacing[diff_idx] / isi)-1

                for i in np.arange(n_samples_skipped):
                    padded_signal = np.insert(padded_signal, diff_idx, np.nan)

            


            # Get indices of timestamps that currently start and end the slice window.
            slice_start_idx = np.where(self.processed_dataframe.index==timestamps[0])[0][0] # np.where returns an array within a tuple. [0][0] gets the value.
            slice_end_idx = np.where(self.processed_dataframe.index==timestamps[-1])[0][0]


            # In all likelihood, these timestamps will be a little offset from what we want. 
            # Calculate the current offset. 

            # I don't remember what any of this was.
            # # How do we get the current zero index when we're working from the padded signal? We can start off by getting signal closest to TTL: 
            # time_closest_to_zero = timestamps[np.argmin(abs(timestamps - time))]
            # sig_at_zero = self.processed_dataframe.loc[time_closest_to_zero, signal_to_slice]
            # # Now we know the value of the signal at 0. 
            # # We don't know for sure that 0 is the only time that the signal is at this value. So, we grab all instances, and their timestamps
            # zero_candidate_indices, = np.where(padded_signal==sig_at_zero)
            # if zero_candidate_indices.size>1:
               #  zero_candidates_timestamps = self.processed_dataframe[self.processed_dataframe[signal_to_slice]==sig_at_zero].index
               #  current_zero_index = zero_candidate_indices[np.where(zero_candidates_timestamps==time_closest_to_zero)[0]][0]            

            current_zero_index = np.argmin(abs(timestamps-time))

            zero_offset = target_zero_index - current_zero_index # Distance of the above from where we want it. 
                # The sign of zero_offset will indicate the necessary direction of the frame shift. 
                # The absolute magnitude of zero_offset indicates the number of frames to shift by. 

            # If target_zero_index is LESS than current_zero_index, a number of elements must be removed from the baseline to shift 
                # the window to the right.  
            if zero_offset < 0:
                padded_signal = np.delete(padded_signal, range(0, abs(zero_offset)))
            # If target_zero_index is greater than the current_zero_index, then the current window must be shifted to the left
                # That is, a number of elements must be added to baseline 
            elif zero_offset > 0:
                timestamps_to_prepend = self.processed_dataframe.index[slice_start_idx - zero_offset : slice_start_idx]
                values_to_add = self.processed_dataframe.loc[timestamps_to_prepend, signal_to_slice]
                padded_signal = np.insert(padded_signal, 0, values_to_add)


            slices.append(padded_signal)


        # Now everything is lined up, but that doesn't mean that it's all the same length. 

        # We've gotten all the baselines lined up, so now it's just a matter of setting the epoch lengths right. 
        # We'll do this by lopping off any extra from the end. 
        min_slice_length = min(s.size for s in slices)
        self.trial_data[reference_TTL] = np.empty([len(slices), min_slice_length])

        for i, s in enumerate(slices):
            self.trial_data[reference_TTL][i] = np.delete(s, range(min_slice_length, s.size))


    def shuffle_align(self, reference_TTL = 'TTL_01', n_iterations = 1000, baseline_time = 10, epoch_time=10, signal_to_slice='Z_Signal'):
        ''' 
            This function will randomly select a number of timepoints equivalent to the number of TTLs registered
            and apply align_to_ttls to those timepoints. It will compute and store the average trace generated by this procedure. 
            It will perform this n_iterations number of times. 

            param self:                 Attributes of signal_processing_object
            param n_iterations:         The number of iterations for which to produce a shuffled alignment
            param baseline_time:        The baseline length to pass to align_to_TTLs
            param epoch_time:           The epoch length to pass to align_to_TTLs
            param signal_to_slice:      The signal ('Z_Signal' or 'RawSignal') which align_to_TTLs is to align
            create self.shuffle_means:  An array in which each row contains the average trace of a set of shuffled data. 
        '''
        try:
            n_ttls = self.ttl_starts[reference_TTL].size
        except (AttributeError, KeyError) as e:
            # If the provided reference TTL has not yet been identified, ask user to run identify_ttls.
            raise Exception(f"You can't shuffle the data based on {reference_TTL} without having run identify_TTLs for {reference_TTL}. Run identify_TTLs first.")

        # Determine the number of bins to compare sliced data against (this is prevent the selection of timestamps from too 
        #    close to either end of the trial)
        try:
            bins = self.trial_data[reference_TTL].shape[1]
        except (AttributeError, KeyError) as e:
            raise Exception(f"I'm not going to let you run these shuffles before generating trial_data for {reference_TTL}. Why risk guessing the bin size? Run align_to_TTLs and then try again.") from e           

        # Now generate the shuffled data.
        self.shuffle_means = np.empty([n_iterations, bins]) # Array for storing output
        shuffled_ttl_name = f'shuffled_{reference_TTL}'
        for i in range(n_iterations):

            # A sort of do/while loop here. Only exit the loop if a mean_signal is generated that will fit in self.shuffle_means 
            # Timestamps within baseline_time from the beginning of the session or epoch_time from the end will produce data with fewer than n_bins
            generate_ttls = True
            while generate_ttls: 
                # Assume that it will get it right right away and prepare to exit loop. 
                generate_ttls = False
                
                # Select the random timestamps and align the data.
                rand_ttls = random.sample(list(self.timestamps), n_ttls)
                self.ttl_starts[shuffled_ttl_name] = rand_ttls
                self.align_to_TTLs(baseline_time = baseline_time, reference_TTL = shuffled_ttl_name, epoch_time=epoch_time, signal_to_slice=signal_to_slice)

                # Take the average and check whether it looks good.
                mean_signal = np.nanmean(self.trial_data[shuffled_ttl_name], axis=0)
                if mean_signal.size < bins:
                    # Try the procedure again if the data aren't the right size
                    generate_ttls = True
            
            # Ensure that when we get to the next iteration of the larger for loop, we'll generate new TTLs
            generate_ttls = True
            
            # Store the data
            normed = calc_robust_z(mean_signal) 
            self.shuffle_means[i] = normed[:bins]
