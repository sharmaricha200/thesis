# CINeMA
This repository contains software that can be used to automate the manual review normally required when performing a non-targeted analysis associated with GCxGC/TOF-MS. CINeMA.py is a python script that automatically compares a list of suggested analytes in a sample to hits from a NIST library search using a user-specified algorithmic or machine learning model. This script will then output a tsv file classifying those hits as high or low matches as well as a pdf file with mirror plots of the suggested analyte's and library hit's mass spectra. In addition to CINeMA.py, under the automation folder there is also chromaTOF_auto.py, a python script that will navigate LECO’s ChromaTOF software's GUI to download the sample's data.

## Installation
* Download the repository and extract all files to the preferred location.
	* The data subdirectory contains optional example data that can be removed.

## CINeMA.py Prerequisites
* Python 3 with the following modules installed:
	* docopt (version 0.6.2)
	* keras (version 2.3.1)
	* matplotlib (version 3.1.1)
	* numpy (version 1.17.3)
	* pandas (version 0.25.2)
	* tensorflow (version 2.0.0)

## chromaTOF_auto.py Prerequisites

* LECO’s ChromaTOF software
* Python 3 with PyAutoGUI (version 0.9.48) installed.
* Text Editor or Python IDE

## Data Directory Structure

*Note: See the data folder provided for an example of how the directories should be formatted.*

* When making predictions with either model, the sample directory needs to contain the list of suggested analytes named “peak_true.msp” and a subdirectory named “hits”. This subdirectory should contain the library hit files in msp format. These files must be named 1.msp, 2.msp, 3.msp, and so on.

* When testing the accuracy of either model, the sample directory needs to contain “peak_true.msp”, the “hits” subdirectory, and “ground_truth.tsv” which is the manual classification of the suggested analyte and library hit match.

* When training a new model, the data directory can contain multiple different sample subdirectories, and each one must contain “peak_true.msp”, the “hits” subdirectory, and “ground_truth.tsv”.

## Command Line Usage

	CINeMA.py -h 

	CINeMA.py ml (train|test|predict) -d <data directory path> -s <path to model> [--e 		<number of epochs>]

	CINeMA.py algo -d <data directory path> (test|predict) [--st (similarity threshold)] 	[--pt <percent threshold>]

	CINeMA.py scatter -d <path to csv>

## Required Arguments:

	ml          			    
		Uses a machine learning model to classify mass spectra

	algo                                	    
		Uses an algorithmic model to classify mass spectra

	scatter
		Generates plots of compound molecular weight to retention times 1 and 2. The CSV must contain column headers labeled “1st 			Retention Time” and “2nd Retention Time” (Default if done through ChromaTOF”

	train				    
		Train a new machine learning model with new data

	test                                               
		Determines the model’s accuracy compared to a manual review

	predict                                         
		Classifies hits as high or low matches

## Optional arguments:
	-h | --help                                              
		Show usage options on the screen

	--st <similarity threshold>                      
		Set the similarity threshold [default: 600]

	--pt <percent threshold>                         
		Set the percent threshold [default: 80]

	--e <Number of epochs>                         
		Set the number of epochs [default: 30]
                           
	--version                                                  
		Show the version number

## Example Usage

*Note: “python” represents the path to your python installation.*

* From command line make the directory where CINeMA.py is located your working directory, then use the following commands.
 
		python CINeMA.py ml predict -d data/algo/sample2 -s model.h5

		python CINeMA.py algo -d data/algo/sample2 predict

		python CINeMA.py rt -d data/algo/sample2/EPA Peaks With Molecular Weight and Formula.csv 

## chromaTOF_auto.py Usage

1. Open up the compound table in ChromaTOF

2. Open ChromaTOF_auto.py in a text editor or IDE.

3. At the bottom of the file, input the number of compounds whose spectra are to be analyzed when calling the export_peaktable function.

	* Note: 10000 is the default value. Ex) export_peaktable(#compounds)

4. Run the script via the IDE or from command line by making the directory where chromaTOF_auto.py is located your working directory, and use the following command:

		python chromaTOF_auto.py

5. Navigate back over to the ChromaTOF window with the table of compounds and let the script run to completion.

	* If you wish to exit the program early, quickly move the mouse up to the left hand of the screen.
