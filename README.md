# OperantBehaviorAnalysis-Python
This repository will be used in the Nautiyal Lab for analyzing output from the Doric Fiber Photometry system.

## Background
The Doric Fiber Photometry system saves both signal and TTL data in large comma-separated-value files as time series data. 
These data can be analyzed in many different ways, as photometry can be applied to many different types of questions about neural activity. 
The goal of this repository is to maintain a simple, user-friendly set of tools for flexibly analyzing photometry data. 

## Usage
The central tool used in this repository is the signal_processing_object class defined within Doric_Processing_Toolbox. This class reads in 
the raw data from a Doric .csv output file and performs a few initial preprocessing steps (isosbestic filtration, artifact removal, etc.). 
The class also contains various methods that I commonly use during analysis (e.g. dff calculation, TTL alignment, etc.). Further, the Toolbox 
contains a set of utility functions that I found helpful to have outside of the class. 

## User Caveats
The initial reading of Doric data-files is, as of this writing in v1.1.1, based on a hard-coded assignment of which hardward inputs corresponds to which channels (excitation vs. isosbestic). 
Not only are these based on the kind of experiment that I'm running (1 excitation signal, 1 isosbestic for 1 fluorophore),
but also they are based on how I have my hardward set up. They should be verified/altered for the user's particular case. 
Future versions may seek to add some flexibility to these as my particular use changes, but for now (v.1.1.1) it's mostly hard-code. 

## Version control
A CHANGELOG is included at the top of the main file, Doric_Processing_Toolbox.py. The version will be updated according to common python practices 
and is also incorporated into the signal_processing_object class, thereby allowing the version of pickled objects to be recorded. 
All writing and debugging, as of v1.1.1 have occurred in python3.8. 
