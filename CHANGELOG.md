#### Version 2.4.1 ####
	-  Oct 25, 2023: Changed function used to load pickled files in signal_processing_object() from pickle.load() to pandas.read_pickle() to provide backwards compatibility with previous versions of pandas. Additionally, signal_processing_object.z_norm_deltaff() was altered to directly return the z normalized signal if there is no window passed. Previously, it had tried to set the window_size to the length of the trial. However, if this length was rounded down, then it would return a nan at the end of the normalized signal.
#### Version 2.4.0 ####
	- Oct 23, 2023: Updated id_transients to use a lowpass (1Hz) filtered signal (4th order) in the calculation of the normalized derivative for the purposes of peak detection.
#### Version 2.3.3 ####
	- # Jun 12, 2023: Addressed corner case in calc_dff_from_percentile in which the length of the signal actually was perfectly divisible by the chosen window length.
#### Version 2.3.2 ####
	- Jun 9, 2023:  Added input variables to calc_dff_from_percentile to enable user to set window size and reference percentile. Defaults are 10s and 5%. 
#### Version 2.3.1 ####
	- May 3, 2023:  There were a few issues identified with calc_dff_from_percentile. First, there were a few bugs in the execution. Second, once those bugs were fixed, it was determined that attempting to use a monototonic linear fit to the 5th percentile is hugely distortionary and simply a bad idea. It has been altered so that the 5th percentile itself in sliding 10s windows is used as F0.
#### Version 2.3.0 ####
	- Apr 10, 2023: Incorporated alternate method for calculated dF/F based on 5th percentile. New function name: calc_dff_from_percentile. Previous function (calc_dff) RENAMED to calc_dff_from_isosbestic. I wouldn't consider this a breaking change because existing data structures aren't modified, but existing scripts will have to be adjusted. 
#### Version 2.2.3 ####
	- Nov 08, 2022: There is occasionally a small amount of drift between the timestamps for the isosbestic and excitation signal. I had placed the tolerance threshold for that at 0.01 seconds, but that was arbitrary. I encountered a data file that drifted 0.012 seconds by the end, so I've bumped the tolerance up to 0.05 seconds. This is also arbitrary. 
#### Version 2.2.2 ####
	- Oct 14, 2022: Finished updating shuffle_align. Changed default reference TTL from "TTL_1" to "TTL_01" and removed re-alignment from end. With ttl_starts as dictionary, re-alignment is not necessary. align_to_ttls was wonky as well. Identifying the zero-index was simplified. 
#### Version 2.2.1 ####
	- Oct 13, 2022: DeltaFF calculation was incorrect. The calculation of the "fitted isosbestic" was erroneously using the excitation signal instead of the isosbestic. Thus, the deltaFF was, essentially, the 465 relative to itself. This has been corrected. Also changed default normalization window in z_norm_deltaff to None. A few minor modifications to the syntax of calc_robust_z. Cosmetic.  
#### Version 2.2.0 ####
	- Oct 04, 2022: Doric provides a set of functions built on the h5py package so that .doric files can be read directly. The multiple file handling has been replaced with these functions. However, the structure of processing .csv vs. an alternate that was introduced in 2.1.0 remains helpful so, instead of creating a new branch the hpy5 functionality is simply being added here. 
#### Version 2.1.1 ####
	-  Sep 28, 2022: Debugged identify_ttls. It ran into trouble when checking for missing data if one of the TTLs was empty. Some of the artifact removal defaults were based on a fixed sampling rate of 12 kSpS. The new software allows this to be modified. As such, this process was made dynamic, and a step was added to ensure that the added buffer doesn't go past the end of the timestamps. 
#### Version 2.1.0 ####
	-  Sep 23, 2022: Added handling of data spread across multiple input files. signal_processing_object.__init__() now has a new boolean variable "single_input" that is set to a default value of True. If multiple files are passed as a dictionary in place of input_file, single_input should be set to False and the data will be more or less processed according to the previous steps from there. Unrelated to the above, I also added the ability to bypass the sliding window in DFF z normalization. 
#### Version 2.0.2 ####
	- Updated calc_robust_z to identify and handle arrays as input.
#### Version 2.0.1 ####
	- Debugging new arguments. Fixed incorrect single-line, multiple-error catching in shuffle_align. Fixed incorrect calculation of first TTL time in identify_ttls when (1.) Data have been dropped during __init__ and (2.) The first TTL starts in the first frame of the new dataframe.
#### Version 2.0.0 ####
	- Edited __init__, identify_TTLs, align_to_TTLs, and shuffle_align methods in signal_processing_object to work with multiple TTLs. All ttl associated data are now stored in a dictionary, associated with key "TTL_1", "TTL_2", ... etc. The latter 3 methods also now take the desired TTL as a keyword argument, with TTL_1 as the default. 
               
    THIS IS A BREAKING CHANGE FROM THE OLD WAY OF HANDLING TTLS. 
#### Version 1.2.2 ####
	- Jun 14, 2022: Updated initial file reading in signal_processing_object.__init__ to support skipping bad lines. Note that the syntax for this function is updated in pandas 1.3.0 and above. I'm currently running 1.2.1 (coincidentally), so I've used the old syntax
#### Version 1.2.1 ####
	- May 26, 2022: Tweaked id_transients() to handle when first event begins at very beginning of session.
#### Version 1.2.0 ####
	- May 25, 2022: Updated calc_robust_z() to accept criteria for baseline normalization. Updated signal_processing_object.z_norm_deltaff to perform normalization based on a sliding window (default 40s based on Liston et al., 2020). Added transient identification function id_transients() based on Liston et al., 2020 as well.
#### Version 1.1.1 ####
	- May 19, 2022: Debugged shuffle_align. There were some unused variables that were holdovers from testing the logic out in a notebook.
#### Version 1.1.0 ####
	- May 19, 2022: Adding ability to read in pickled signal_processing_objects for easier integration of future updates. Version tracking has also been added to signal_processing_object.__init__ using the global variable __version__ defined at the end of the changelog. ALWAYS UPDATE __version__ AFTER RECORDING A CHANGE.
