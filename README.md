# Spatial-Short-Term-Average (SSTA)
Summary:

SSTA is a method for detecting nanoseismicity within DAS data with potential applications in the detection of events 
within spatially dense data.  This repo contains the code and data to reproduce the results in the paper 
(submission pending). The attached notebook takes you through a step-by-step guide for using this methods to process 
DAS data to detect nanoseismicity using the SSTA and other potential methods such as the STA/LTA, kNN and Amplitude 
Thresholding. Any questions or problems using the code contact me at: kcx19zeu@uea.ac.uk.

Packages Needed:

    - matplotlib
    - tdqm
    - obspy
    - numpy
    - pickle
    - sklearn
    - pandas

Code:
    
    - Event Detection Comparison.ipynb
    - SSTA.py
    - clusterAndClassify.py
    - filters.py
    - TDMS_Read.py
    - TDMS_Utilities.py

Data:

    -Earthquake Data
        - Cromer_Earthquake_UTC_20250126_043223.140.tdms
        - Cromer_Earthquake_UTC_20250126_043253.140.tdms
        - Cromer_Earthquake_UTC_20250126_043323.140.tdms
    -Example Windows
        - February_Window_UTC_20240227_103823.084.tdms
        - January_Window_UTC_20240117_130137.656.tdms
        - November_Window_UTC_20231109_134947.573.tdms
    -kNN Anoms
        - NDay
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3
        - JDay
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3
        - FDay
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3
        - NDay Filtered
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3
        - JDay Filtered
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3
        - FDay Filtered
            - kNNAnomsTest
            - kNNAnomsTest2
            - kNNAnomsTest3