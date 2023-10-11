# Machine learning technology for EEG-prognosis of opening of the blood-brain barrier and the activation of brain’s drainage system during isoflurane anesthesia

## Description of included data
There are two examples of data used for training and data of two rats with isoflurane anestesia.
* The data of file "1 rat 1 before the opening BBB.edf" from 1800 sec. to 3600 sec. were used as data of normal behaviour marked as '0' during the training process
* The data of file "2 rat 1 after the opening BBB.edf" from 1800 sec. to 3600 sec. were used as data of opened BBB marked as '1' during the training process.

The application of trained ANN was provided for two next files of EEG recorded in rats with 1% and 4% isoflurane anesthesia.  
* "Rat_2.4.edf"  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; without anesthesia – 0 sec.  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; anesthesia 1% – starting from 2400 sec.  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; anesthesia 4% (euthanasia) – starting from 4800 sec.  
* "Rat_2.11.edf"  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; without anesthesia – 0 sec.  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; anesthesia 1% – starting from 1800 sec.  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; anesthesia 4% (euthanasia) – starting from 3600 sec.  

## Brief description of program files
### fft_diap.py
First, we calculate the corresponding spectral characteristics for five frequency bands:  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; gamma = (30,50)  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; beta = (12,30)  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; alpha = (8,12)  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; theta = (4,8)  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; delta = (1,4)  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; low = (0,1)  
The numbers are in Hz.  
This program gets one edf-file and provides the temporal implementations of five band powers in a sliding window. Finally, we get this in a '.csv' file of the same name as was in '.edf'.

### ann_BBB.py
This program trains the ANN by different '.csv' obtained for rats with 1 -- opened BBB and 0 -- normal behaviour. After processing we get a folder "ANN" with trained ANN in a keras format.

### ann_validation_box.py
This program gets the trained ANN from the folder "ANN" (provided in a previous program). Then we apply it to the new data of rats with anesthesia and calculate the statistics of detecting the activation of lymphatic drainage function or opening the BBB.

