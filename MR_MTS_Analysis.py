# Title: Analysis of the MR test results, obtained from the MTS machine in EB.
#
# Author: Farhad Abdollahi (abdolla4@msu.edu) under supervision of Prof. Kutay (kutay@msu.edu)
# Date: 11/14/2023
# Update: 05/08/2024 - Tab delimiter files, cycle recognition based on Force Command.
# ======================================================================================================================

# Importing the required libraries.
import os
import json
import pickle
import openpyxl
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from scipy.optimize import minimize
import tkinter as tk
from tkinter import filedialog, messagebox
from time import perf_counter


def main():
    """
    This is the main function to run this code.
    :return:
    """
    print(f'=============================================================================================')
    print(f'=============================================================================================')
    print(f'============ Software for analyzing the MR test results from MTS device in EB ===============')
    print(f'=== Developed by Farhad Abdollahi (abdolla4@msu.edu) and Tanzila Islam (islamta1@msu.edu) ===')
    print(f'=== under supervision of Prof. Kutay (kutay@msu.edu).                                     ===')
    print(f'=== Latest update at: 11/15/2023 (05/08/2024) (08/01/2024)                                ===')
    print(f'=============================================================================================')
    print(f'=============================================================================================\n')

    # Asks the input path to the CSV file for the analysis.
    root = tk.Tk()
    root.withdraw()
    InputFile = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")])

    # Check if the file is existed.
    if not os.path.isfile(InputFile):
        raise FileNotFoundError(f'The selected file is not found: {InputFile}')

    # Otherwise, start the program.
    # First, analyze the first line to recognize the delimiter and column names.
    cont = []
    with open(InputFile, 'r') as file:              # Read the first 20 lines.
        for i, line in enumerate(file):             # read the file line by line.
            cont.append(line)
            if i == 20:
                break
    for i in range(1, len(cont)):
        if (cont[i] == '\n' and cont[i-1] == '\n') or \
                (cont[i].replace('"', '') == '\n' and cont[i-1].replace('"', '') == '\n') or \
                (cont[i].replace(',', '') == '\n' and cont[i-1].replace(',', '') == '\n'):
            Columnline  = cont[i+1].replace('\n', '')       # Get the column names.
            dataline    = cont[-1].replace('\n', '')        # Get a data line.
            SkipRows    = i + 3                             # Number of rows to skip.
            break
    # Determine the tab or comma delimiter structure of the data.
    if ',' in dataline:
        Delimiter = ','
    elif '\t' in dataline:
        Delimiter = '\t'
    else:
        raise Exception('Data file seems to be neither comma or tab delimiter. Please check the file.')
    # Determine the column names.
    NameMatchingRatio   = lambda name1, name2: fuzz.ratio(name1.lower(), name2.lower())
    Columnline          = Columnline.split(Delimiter)
    ColNames            = [str(i) for i in range(len(Columnline))]
    ColMatchRatios      = [0.0 for i in range(len(Columnline))]
    AlternativeNames    = ['Cycle Count', 'RunTime', 'Disp', 'Force', 'Command', 'Time']
    MatchRatios         = np.zeros((len(Columnline), len(AlternativeNames)))
    for i in range(len(Columnline)):
        for j in range(len(AlternativeNames)):
            MatchRatios[i, j] = NameMatchingRatio(Columnline[i], AlternativeNames[j])
    for i in range(len(AlternativeNames)):
        idx = np.argmax(MatchRatios[:, i])
        if MatchRatios[idx, i] > ColMatchRatios[idx]:
            ColNames[idx] = AlternativeNames[i]
            ColMatchRatios[idx] = MatchRatios[idx, i]
    for i in range(len(ColNames)):
        if ColNames[i] == 'Cycle Count':
            ColNames[i] = 'Cycle'
    # Now, read the CSV file.
    data = pd.read_csv(InputFile, skiprows=SkipRows, names=ColNames, delimiter=Delimiter)
    print(f'* CSV file has been read. There are <<{len(data)}>> rows of data.')

    # Calculate the range of each cycle. This parameter can be very helpful in recognition of the test sequences.
    if 'Command' in data.columns:
        CycleInfo = Calc_Cycle_Info_Command(data)
    else:
        CycleInfo = Calc_Cycle_Info(data)
    # Calculate the MR at different sequences.
    CycleInfo, SeqInfo, Diameter, Height = Calc_MR(CycleInfo)

    # Evaluate the PD.
    PDResult = Calc_PD(CycleInfo, SeqInfo, Height, InputFile)

    # Export the data in a form of an excel file.
    if PDResult == None:
        ExportData2Excel(InputFile, CycleInfo, SeqInfo)
    else:
        ExportData2Excel_PD(InputFile, CycleInfo, SeqInfo, PDResult)

    # Return Nothing.
    return
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def Calc_Cycle_Info(data):
    """
    This function calculates some information about each cycle.
    :param data: A DataFrame which includes test results.
    :return: A new DataFrame with the cycle information.
    """
    st          = perf_counter()
    Cycles      = data['Cycle'].unique()        # Get an array of all cycle numbers that the MTS provides.
    LenCycles   = len(Cycles)

    # This dictionary is for storing the information about each cycle.
    Res = {
        'SequenceNum'   : [],
        'ActualCycle'   : [],
        'ForceRange'    : [],
        'DispRange'     : [],
        'StartTime'     : [],
        'EndTime'       : [],
        'Duration'      : [],
        'StartCycle'    : [],
        'EndCycle'      : [],
        'AvgForceNoise_kN': [],
        'AvgForceNoise_%': [],
        'PeakForceDiff_kN': [],
        'PeakForceDiff_%' : [],
        'RestDisp'        : []
    }

    # Now, iterate on these cycles to find the loading patterns.
    #   NOTE: MTS defines three cycles for each actual loading cycles: loading, unloading, rest. In this loop, we need
    #   to recognize these cycles.
    AvgNumData, NumCycles = 0, 0        # Two variables to store the average number of datapoints in each load cycle.
    MTScyc = 0                          # Index counter for the MTS cycles.
    ActualCycle = 1                     # Counter for actual loading pattern cycles.
    SeqNum = 0                          # Index of the current loading sequence.
    SeqCycleCounter = 0                 # A counter for number of cycles in each sequence.
    SeqForceRangeAvg = 0                # A variable to store the avg force range in each sequence.
    while MTScyc < LenCycles - 3:
        # Get the slopes of the lines for the next three MTS cycles.
        Slopes = np.zeros(3)
        for i in range(3):
            Time  = data[data['Cycle'] == Cycles[MTScyc+i]]['RunTime'].to_numpy()
            Force = data[data['Cycle'] == Cycles[MTScyc+i]]['Force'].to_numpy()
            slope, _ = np.polyfit(x=Time, y=Force, deg=1)
            Slopes[i] = slope

        # Check if the slopes are following some pattern of load/unload/rest.
        CheckRestPart = np.abs(Slopes[2]) < 0.20 * np.abs(Slopes[:2]).mean()
        CheckLoadUnloadSign = np.sign(Slopes[0]) != np.sign(Slopes[1])
        if CheckRestPart and CheckLoadUnloadSign:
            # Means we have a loading pattern cycle.
            Time  = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc+3])]['RunTime'].to_numpy()
            Force = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc + 3])]['Force'].to_numpy()
            Disp  = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc + 3])]['Disp'].to_numpy()
            # Now, extract the information of these cycles.
            StartTime = Time[0]
            EndTime   = Time[-1]
            Duration  = EndTime - StartTime
            ForceRange= Force.max() - Force.min()
            DispRange = Disp.max() - Disp.min()
            RestDisp  = data[data['Cycle'] == Cycles[MTScyc+2]]['Disp'].mean()
            # Check for the end of the sequence.
            if SeqCycleCounter < 90:        # at least 100 cycles is expected. For less than 90, just assume the same.
                SeqForceRangeAvg = ((SeqForceRangeAvg * SeqCycleCounter) + ForceRange) / (SeqCycleCounter + 1)
                SeqCycleCounter += 1
            else:
                if np.abs(ForceRange - SeqForceRangeAvg) > (0.25 * SeqForceRangeAvg):
                    # Go to the next sequence and reset every thing.
                    ActualCycle = 1             # Actual cycle count is resetted.
                    SeqNum     += 1             # Sequence number is increased.
                    SeqForceRangeAvg = 0        # Reset the force range and counter of cycles in each sequence.
                    SeqCycleCounter  = 0
                else:
                    SeqForceRangeAvg = ((SeqForceRangeAvg * SeqCycleCounter) + ForceRange) / (SeqCycleCounter + 1)
                    SeqCycleCounter += 1
            # Save the results for this cycle.
            Res['ActualCycle'].append(ActualCycle)
            Res['ForceRange'].append(ForceRange)
            Res['DispRange'].append(DispRange)
            Res['StartTime'].append(StartTime)
            Res['EndTime'].append(EndTime)
            Res['Duration'].append(Duration)
            Res['StartCycle'].append(Cycles[MTScyc])
            Res['EndCycle'].append(Cycles[MTScyc + 2])
            Res['SequenceNum'].append(SeqNum)
            Res['AvgForceNoise_kN'].append(None)
            Res['AvgForceNoise_%'].append(None)
            Res['PeakForceDiff_kN'].append(None)
            Res['PeakForceDiff_%'].append(None)
            Res['RestDisp'].append(RestDisp)
            # Update the indices and parameters.
            MTScyc += 3         # Updating the index of MTS cycle counter.
            ActualCycle += 1    # Updating the actual cycle number.
            AvgNumData = ((AvgNumData * NumCycles) + len(Time)) / (NumCycles + 1)
            NumCycles += 1

        else:       # Means we didn't recognize a cycle.
            # Check if it is the start of the loading sequence, so keep swiping the data to find the loading cycles.
            if AvgNumData == 0:
                MTScyc += 1
                continue
            else:
                # Otherwise, let's check for the pausing breaks between the loading sequences (check by size).
                NumDataPoints = np.array([len(data[data['Cycle'] == Cycles[MTScyc + i]]) for i in range(3)])
                if NumDataPoints.sum() > (5 * AvgNumData):
                    # Means there is a break happening, so update the indices and every thing.
                    MTScyc += 1 + np.argmax(NumDataPoints)      # Updating the index of MTS cycles.
                    ActualCycle = 1                             # Resetting the actual cycle numbers.
                    AvgNumData, NumCycles = 0, 0                # Resetting the avg number of data in each cycle.
                    SeqNum += 1                                 # Go to the next sequence.
                    SeqForceRangeAvg = 0                        # Reset the force range and counter of cycles in
                    SeqCycleCounter  = 0                        #   each sequence.
                else:
                    # Maybe we have a noise.
                    # Let's skip for now.
                    MTScyc += 3
                    ActualCycle += 1
                    NumCycles += 1
                    # raise Exception("We don't know what is happening :) you need to run the code in Debug mode and stop at this line to look for the problem.")

        # Pring the progress.
        print(f'* Analyzing the CSV file, please wait. Progress: {(MTScyc + 3) / LenCycles * 100:.2f}%, '
              f'[Passed time: {perf_counter() - st:.2f} sec.]', end='\r')

    # Create a DataFrame from the Results.
    ResDF = pd.DataFrame(Res)
    print(f'\n* CSV file has been analyzed:')
    print(f'\t-- Number of loading cycles recognized:\t\t{len(ResDF)}')
    print(f'\t-- Number of loading sequences recognized:\t{len(ResDF["SequenceNum"].unique())}')

    # Return the dataframe.
    return ResDF
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def Calc_Cycle_Info_Command(data):
    """
    This function calculates some information about each cycle.
    :param data: A DataFrame which includes test results.
    :return: A new DataFrame with the cycle information.
    """
    st          = perf_counter()
    Cycles      = data['Cycle'].unique()        # Get an array of all cycle numbers that the MTS provides.
    LenCycles   = len(Cycles)

    # This dictionary is for storing the information about each cycle.
    Res = {
        'SequenceNum'   : [],
        'ActualCycle'   : [],
        'ForceRange'    : [],
        'DispRange'     : [],
        'StartTime'     : [],
        'EndTime'       : [],
        'Duration'      : [],
        'StartCycle'    : [],
        'EndCycle'      : [],
        'AvgForceNoise_kN': [],
        'AvgForceNoise_%' : [],
        'PeakForceDiff_kN': [],
        'PeakForceDiff_%' : [],
        'RestDisp'        : []
    }

    # Now, iterate on these cycles to find the loading patterns.
    #   NOTE: MTS defines three cycles for each actual loading cycles: loading, unloading, rest. In this loop, we need
    #   to recognize these cycles.
    AvgNumData, NumCycles = 0, 0        # Two variables to store the average number of datapoints in each load cycle.
    MTScyc = 0                          # Index counter for the MTS cycles.
    ActualCycle = 1                     # Counter for actual loading pattern cycles.
    SeqNum = 0                          # Index of the current loading sequence.
    SeqCycleCounter = 0                 # A counter for number of cycles in each sequence.
    SeqForceRangeAvg = 0                # A variable to store the avg force range in each sequence.
    while MTScyc < LenCycles - 3:
        # Get the slopes of the lines for the next three MTS cycles.
        Slopes = np.zeros(3)
        for i in range(3):
            Time    = data[data['Cycle'] == Cycles[MTScyc+i]]['RunTime'].to_numpy()
            Force   = data[data['Cycle'] == Cycles[MTScyc+i]]['Force'].to_numpy()
            Command = data[data['Cycle'] == Cycles[MTScyc+i]]['Command'].to_numpy()
            slope, _ = np.polyfit(x=Time, y=Command, deg=1)
            Slopes[i] = slope

        # Check if the slopes are following some pattern of load/unload/rest.
        CheckRestPart = np.abs(Slopes[2]) < 0.20 * np.abs(Slopes[:2]).mean()
        CheckLoadUnloadSign = np.sign(Slopes[0]) != np.sign(Slopes[1])
        if CheckRestPart and CheckLoadUnloadSign:
            # Means we have a loading pattern cycle.
            Time    = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc+3])]['RunTime'].to_numpy()
            Force   = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc + 3])]['Force'].to_numpy()
            Command = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc + 3])]['Command'].to_numpy()
            Disp    = data[data['Cycle'].isin(Cycles[MTScyc:MTScyc + 3])]['Disp'].to_numpy()
            # Now, extract the information of these cycles.
            StartTime       = Time[0]
            EndTime         = Time[-1]
            Duration        = EndTime - StartTime
            ForceRange      = Force.max() - Force.min()
            DispRange       = Disp.max() - Disp.min()
            CommandRange    = Command.max() - Command.min()
            AvgForceNoise   = np.abs(Command - Force).mean()
            AvgForceNoisep  = AvgForceNoise / CommandRange * 100
            PeakForceDiff   = np.abs(CommandRange - ForceRange)
            PeakForceDiffp  = PeakForceDiff / CommandRange * 100
            RestDisp        = data[data['Cycle'] == Cycles[MTScyc+2]]['Disp'].mean()
            # Check for the end of the sequence.
            if SeqCycleCounter < 90:        # at least 100 cycles is expected. For less than 90, just assume the same.
                SeqForceRangeAvg = ((SeqForceRangeAvg * SeqCycleCounter) + ForceRange) / (SeqCycleCounter + 1)
                SeqCycleCounter += 1
            else:
                if np.abs(ForceRange - SeqForceRangeAvg) > (0.25 * SeqForceRangeAvg):
                    # Go to the next sequence and reset every thing.
                    ActualCycle = 1             # Actual cycle count is resetted.
                    SeqNum     += 1             # Sequence number is increased.
                    SeqForceRangeAvg = 0        # Reset the force range and counter of cycles in each sequence.
                    SeqCycleCounter  = 0
                else:
                    SeqForceRangeAvg = ((SeqForceRangeAvg * SeqCycleCounter) + ForceRange) / (SeqCycleCounter + 1)
                    SeqCycleCounter += 1
            # Save the results for this cycle.
            Res['ActualCycle'].append(ActualCycle)
            Res['ForceRange'].append(ForceRange)
            Res['DispRange'].append(DispRange)
            Res['StartTime'].append(StartTime)
            Res['EndTime'].append(EndTime)
            Res['Duration'].append(Duration)
            Res['StartCycle'].append(Cycles[MTScyc])
            Res['EndCycle'].append(Cycles[MTScyc + 2])
            Res['SequenceNum'].append(SeqNum)
            Res['AvgForceNoise_kN'].append(AvgForceNoise)
            Res['AvgForceNoise_%'].append(AvgForceNoisep)
            Res['PeakForceDiff_kN'].append(PeakForceDiff)
            Res['PeakForceDiff_%'].append(PeakForceDiffp)
            Res['RestDisp'].append(RestDisp)
            # Update the indices and parameters.
            MTScyc += 3         # Updating the index of MTS cycle counter.
            ActualCycle += 1    # Updating the actual cycle number.
            AvgNumData = ((AvgNumData * NumCycles) + len(Time)) / (NumCycles + 1)
            NumCycles += 1

        else:       # Means we didn't recognize a cycle.
            # Check if it is the start of the loading sequence, so keep swiping the data to find the loading cycles.
            if AvgNumData == 0:
                MTScyc += 1
                continue
            else:
                # Otherwise, let's check for the pausing breaks between the loading sequences (check by size).
                NumDataPoints = np.array([len(data[data['Cycle'] == Cycles[MTScyc + i]]) for i in range(3)])
                if NumDataPoints.sum() > (5 * AvgNumData):
                    # Means there is a break happening, so update the indices and every thing.
                    MTScyc += 1 + np.argmax(NumDataPoints)      # Updating the index of MTS cycles.
                    ActualCycle = 1                             # Resetting the actual cycle numbers.
                    AvgNumData, NumCycles = 0, 0                # Resetting the avg number of data in each cycle.
                    SeqNum += 1                                 # Go to the next sequence.
                    SeqForceRangeAvg = 0                        # Reset the force range and counter of cycles in
                    SeqCycleCounter  = 0                        #   each sequence.
                else:
                    raise Exception("We don't know what is happening :) you need to run the code in Debug mode and stop at this line to look for the problem.")

        # Pring the progress.
        print(f'* Analyzing the CSV file, please wait. Progress: {(MTScyc + 3) / LenCycles * 100:.2f}%, '
              f'[Passed time: {perf_counter() - st:.2f} sec.]', end='\r')

    # Create a DataFrame from the Results.
    ResDF = pd.DataFrame(Res)
    print(f'\n* CSV file has been analyzed:')
    print(f'\t-- Number of loading cycles recognized:\t\t{len(ResDF)}')
    print(f'\t-- Number of loading sequences recognized:\t{len(ResDF["SequenceNum"].unique())}')

    # Return the dataframe.
    return ResDF
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def Calc_MR(CycleInfo):
    """
    This function calculates the MR using the cycle information.
    :param CycleInfo:
    :return: the updated "CycleInfo" and the new "SeqInfo" dataframes.
    """
    # Inorder to analyze the cycle informations, first the geometry properties of the sample is required from the user.
    Diameter, Height = GetInputFromUser()

    # Now, calculating the deviatoric stress and vertical strain of the sample using the peak to peak values.
    #       Note: the force is in kN, and the displacement is in mm.
    CycleInfo['Stress_kPa'] = CycleInfo['ForceRange'] / (np.pi / 4 * (Diameter * 0.0254) ** 2)      # Stress in kPa.
    CycleInfo['Strain']     = CycleInfo['DispRange'] / (Height * 25.4)                              # Strain.
    CycleInfo['MR_MPa']     = CycleInfo['Stress_kPa'] / CycleInfo['Strain'] / 1e3                   # MR in MPa.
    CycleInfo['MR_psi']     = CycleInfo['MR_MPa'] * 145.038                                         # MR in psi

    # Now, we need to calculate the average of last 5 cycles for calculating the representative MR of each sequence.
    Res = {'Sequence Number': [], 'Deviator Stress (kPa)': [], 'Strain': [],
           'MR (MPa)': [], 'MR (psi)': [], 'Num Cycles': []}
    SeqNums = CycleInfo['SequenceNum'].unique()
    for sq in SeqNums:
        TempDF = CycleInfo[CycleInfo['SequenceNum'] == sq]
        # Now, save the results.
        Res['Sequence Number'].append(sq)
        Res['Deviator Stress (kPa)'].append(TempDF['Stress_kPa'].to_numpy()[-5:].mean())
        Res['Strain'].append(TempDF['Strain'].to_numpy()[-5:].mean())
        Res['MR (MPa)'].append(TempDF['MR_MPa'].to_numpy()[-5:].mean())
        Res['MR (psi)'].append(TempDF['MR_psi'].to_numpy()[-5:].mean())
        Res['Num Cycles'].append(len(TempDF))

    # Generate a dataframe from the results.
    SeqInfo = pd.DataFrame(Res)

    # Print the results to user.
    print(f'* Sequences has been analyzed. The results are as follows:\n')
    SeqInfo2show = SeqInfo.copy()
    SeqInfo2show['Deviator Stress (kPa)']   = SeqInfo2show['Deviator Stress (kPa)'].map('{:.2f}'.format)
    SeqInfo2show['Strain']                  = SeqInfo2show['Strain'].map('{:.3e}'.format)
    SeqInfo2show['MR (MPa)']                = SeqInfo2show['MR (MPa)'].map('{:.2f}'.format)
    SeqInfo2show['MR (psi)']                = SeqInfo2show['MR (psi)'].map('{:.2f}'.format)
    print(SeqInfo2show.to_string(index=False))
    print('\n')

    # Return the dataframe.
    return CycleInfo, SeqInfo, Diameter, Height
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def GetInputFromUser():
    """
    This function is only to get some inputs from the user.
    :return:
    """
    # Define a function to check if the entered value is float.
    def CheckInputValue(value, InvalidTitle):
        while True:
            try:
                value = float(value)
                return value
            except:
                value = input(InvalidTitle)

    # Get the inputs from the user.
    Diameter = input('Please enter the sample diameter (inches):')
    Diameter = CheckInputValue(Diameter, 'You should enter a valid float number. '
                                         'Please enter the sample diameter again (inches):')
    Height   = input('Please enter the sample height (inches):')
    Height   = CheckInputValue(Height, 'You should enter a valid float number. '
                                         'Please enter the sample height again (inches):')

    # Return the values.
    return Diameter, Height
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def GetStressStateFromUser():
    """
    This function is only to get some stress state inputs from the user.
    :return:
    """
    # Define a function to check if the entered value is float.
    def CheckInputValue(value, InvalidTitle):
        while True:
            try:
                value = float(value)
                return value
            except:
                value = input(InvalidTitle)

    # Get the inputs from the user.
    cyclicStrs  = input('Please enter the deviatoric (cyclic) stress during the PD test (psi):')
    cyclicStrs  = CheckInputValue(cyclicStrs, 'You should enter a valid float number. ' 
                                              'Please enter the deviatoric (cyclic) stress during PD test again (psi):')
    Confine     = input('Please enter the confining pressure during the PD test (psi):')
    Confine     = CheckInputValue(Confine, 'You should enter a valid float number. ' 
                                           'Please enter the confining pressure during PD test again (psi):')
    Atmospheric = input('Please enter the atmospheric pressure during the PD test (psi), [e.g. 14.6 psi]:')
    Atmospheric = CheckInputValue(Atmospheric, 'You should enter a valid float number. ' 
                                               'Please enter the atmospheric pressure during the PD test (psi), '
                                               '[e.g. 14.6 psi]:')

    # Return the values.
    return cyclicStrs, Confine, Atmospheric
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def Calc_PD(CycleInfo, SeqInfo, Height, InputFile):
    """
    This function analyze the permanent deformation test results and return the plastic strain growth with the loading
        cycles as well as the stress state during the PD test.
    :param CycleInfo: A DataFrame of all loading cycle responses.
    :param SeqInfo: A DataFrame of sequence informations to find the PD sequence.
    :param Height: The height of the sample to calculate the plastic strain.
    :param InputFile: Path to the input file name for saving the PD results in the same directory.
    :return: A dictionary of all results.
    """
    # Check if there is a PD sequence in the result: PD sequences are usually consists of about 10,000 loading cycles.
    #   However, as the test might encounter failure sooner, we recognize the PD sequence as a seq with at least 600
    #   loading cycles.
    Indx = np.where(SeqInfo['Num Cycles'].to_numpy() > 600)[0]
    if len(Indx) == 0:
        print(f'* Permanent Deformation loading cycles were not found!')
        return

    # Otherwise, calculate the plastic strain growth with cycle.
    PDsq = SeqInfo.loc[Indx[0], 'Sequence Number']
    Cycles = CycleInfo[CycleInfo['SequenceNum'] == PDsq].copy()
    Cycles.reset_index(inplace=True)
    Cycles['PlasticDisp'] = (Cycles['RestDisp'] - float(Cycles.loc[0, 'RestDisp'])) * -1
    Cycles['PlasticStrain'] = Cycles['PlasticDisp'] / Height
    CycleNum = Cycles['ActualCycle'].to_numpy()
    PlasticStrn = Cycles['PlasticStrain'].to_numpy()
    print(f'* Permanent Deformation loading cycles were found!')
    print(f'   ** there are {len(CycleNum)} loading cycles under sequence number {PDsq}')

    # Get the stress state properties.
    DeviatoricStrs, ConfineStrs, AtmStrs = GetStressStateFromUser()

    # Now fit a model to the results.
    # First, Thompson & Neumann (1993): log ep = a + b * log N
    FitCoeff = np.polyfit(np.log10(CycleNum[1:]), np.log10(PlasticStrn[1:]), 1)
    b, a = FitCoeff
    print(f'   ** First, Thompson and Neumann (1993) model: (log εp = a + b * log N)')
    print(f'\t\ta = {a:.6e}\n\t\tb = {b:.6e}')
    # Next, Ullditz equation:
    #
    # print(f'   ** Next, Ullditz (1993) model: (εp = A * (N ^ α) * [σz / σatm] ^ β)')
    # print(f'\t\tA = {A:.6e}\n\t\tα = {b:.6e}\n\t\tβ = {b:.6e}')

    # Aggregate the results in a dictionary.
    Result = {
        'CycleNum': CycleNum.tolist(),
        'PlasticStrn': PlasticStrn.tolist(),
        'AtmStrs': AtmStrs,
        'DeviatoricStrs': DeviatoricStrs,
        'ConfineStrs': ConfineStrs,
        'FitCoeff': {
            'Thompson&Neumann': [a, b]
        }
    }

    # Save that dictionary along with the file.
    PDFileName = os.path.join(os.path.dirname(InputFile),
                              os.path.splitext(os.path.basename(InputFile))[0] + '-PD-Output.json')
    json.dump(Result, open(PDFileName, 'w'))

    # Return the PD results.
    return Result
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def ExportData2Excel(InputFile, CycleInfo, SeqInfo):
    """
    This function exports the data in a form of an excel file.
    :param InputFile: The path to the input CSV file.
    :return: Nothing.
    """
    # Create the path for writing the Excel file.
    ExcelFileName   = os.path.join(os.path.dirname(InputFile),
                                   os.path.splitext(os.path.basename(InputFile))[0] + '-Output.xlsx')

    # And now, write the sequence information in another sheet within that excel file.
    SeqInfo_SI = SeqInfo.copy()
    SeqInfo_US = SeqInfo.copy()
    SeqInfo_US['Deviator Stress (psi)'] = SeqInfo_US['Deviator Stress (kPa)'] * 0.145038
    SeqInfo_US.drop('Deviator Stress (kPa)', axis=1, inplace=True)
    SeqInfo_US.drop('MR (MPa)', axis=1, inplace=True)
    SeqInfo_SI.drop('MR (psi)', axis=1, inplace=True)
    # First add some more rows to the SeqInfo dataframe.
    SeqInfo_SI['Air Pressure (kPa)'] = 102.201  # Use the value for Lansing, MI in Nov 15, 2023
    SeqInfo_US['Air Pressure (psi)'] = SeqInfo_SI['Air Pressure (kPa)'] * 0.145038
    try:
        Confine_AASHTOt307_psi = [15, 3, 3, 3, 5, 5, 5, 10, 10, 10, 15, 15, 15, 20, 20, 20]
        if len(SeqInfo_US) <= len(Confine_AASHTOt307_psi):
            SeqInfo_US['Confine Pressure (psi)'] = Confine_AASHTOt307_psi[:len(SeqInfo_US)]
        else:
            SeqInfo_US['Confine Pressure (psi)'] = Confine_AASHTOt307_psi + \
                                                   [0 for i in range(len(SeqInfo_US) - len(Confine_AASHTOt307_psi))]
    except:
        SeqInfo_US['Confine Pressure (psi)'] = 0
    SeqInfo_SI['Confine Pressure (kPa)'] = SeqInfo_US['Confine Pressure (psi)'] / 0.145038
    SeqInfo_US['σ3 (psi)'] = 0
    SeqInfo_SI['σ3 (kPa)'] = 0
    SeqInfo_US['σ1 (psi)'] = 0
    SeqInfo_SI['σ1 (kPa)'] = 0
    SeqInfo_US['τoct (psi)'] = 0
    SeqInfo_SI['τoct (kPa)'] = 0
    SeqInfo_US['Bulk (psi)'] = 0
    SeqInfo_SI['Bulk (kPa)'] = 0
    SeqInfo_US['Bulk/Pa'] = 0
    SeqInfo_SI['Bulk/Pa'] = 0
    SeqInfo_US['τoct/Pa+1'] = 0
    SeqInfo_SI['τoct/Pa+1'] = 0
    SeqInfo_US['Predict MR (psi)'] = 0
    SeqInfo_SI['Predict MR (MPa)'] = 0
    SeqInfo_US['MR MAE (psi)'] = 0
    SeqInfo_SI['MR MAE (MPa)'] = 0

    # Now writing them in two different sheets, as well as one more sheet for cycle information details.
    with pd.ExcelWriter(ExcelFileName, engine='openpyxl') as writer:
        SeqInfo_US.to_excel(writer, sheet_name='Seq_Info_US', index=False)
        SeqInfo_SI.to_excel(writer, sheet_name='Seq_Info_SI', index=False)
        CycleInfo.to_excel(writer, sheet_name='Cycle_Info', index=False)

    # Now, modifying the sequence information sheets and provide the formulas in the cells.
    Workbook = openpyxl.load_workbook(ExcelFileName)            # Read the excel file with OpenPyXL module.
    SheetUS  = Workbook['Seq_Info_US']                          # Read the US sheet.
    SheetSI  = Workbook['Seq_Info_SI']                          # Read the US sheet.

    # Add a formula to the Cells.
    for i in range(len(SeqInfo_US)):
        SheetUS[f'H{2 + i}'] = f'=G{2 + i}'                     # Fixing the σ3.
        SheetSI[f'H{2 + i}'] = f'=G{2 + i}'
        SheetUS[f'I{2 + i}'] = f'=H{2 + i} + E{2 + i}'          # Fixing the σ1.
        SheetSI[f'I{2 + i}'] = f'=H{2 + i} + E{2 + i}'
        SheetUS[f'J{2 + i}'] = f'=1/3*SQRT((I{2 + i}-H{2 + i})^2+(I{2 + i}-H{2 + i})^2)'    # Fixing τoct.
        SheetSI[f'J{2 + i}'] = f'=1/3*SQRT((I{2 + i}-H{2 + i})^2+(I{2 + i}-H{2 + i})^2)'
        SheetUS[f'K{2 + i}'] = f'=I{2 + i}+H{2 + i}*2'          # Fixing the bulk stress.
        SheetSI[f'K{2 + i}'] = f'=I{2 + i}+H{2 + i}*2'
        SheetUS[f'L{2 + i}'] = f'=K{2 + i} / F{2 + i}'          # Fixing the bulk/Pa ratio.
        SheetSI[f'L{2 + i}'] = f'=K{2 + i} / F{2 + i}'
        SheetUS[f'M{2 + i}'] = f'=J{2 + i} / F{2 + i} + 1'      # Fixing the τoct/Pa+1 ratio.
        SheetSI[f'M{2 + i}'] = f'=J{2 + i} / F{2 + i} + 1'
        SheetUS[f'N{2 + i}'] = f'=($T$4*F{2 + i}*L{2 + i}^$T$5*M{2 + i}^$T$6)'      # Fixing the predicted MR (psi).
        SheetSI[f'N{2 + i}'] = f'=($T$4*F{2 + i}*L{2 + i}^$T$5*M{2 + i}^$T$6)/1000' # MR in MPa
        SheetUS[f'O{2 + i}'] = f'=ABS(N{2 + i} - C{2 + i})'     # Fixing the MR MAE.
        SheetSI[f'O{2 + i}'] = f'=ABS(N{2 + i} - C{2 + i})'
        SheetUS['T7'] = f'=RSQ((N3:N17),(C3:C17))'              # Fixing the R-square values.
        SheetSI['T7'] = f'=RSQ((N3:N17),(C3:C17))'

    # Add the k values.
    k1_US, k2_US, k3_US, k1_SI, k2_SI, k3_SI = Calc_kvalues(SeqInfo_US.copy(), SeqInfo_SI.copy())
    SheetUS['T4'] = f'{k1_US}'
    SheetUS['T5'] = f'{k2_US}'
    SheetUS['T6'] = f'{k3_US}'
    SheetSI['T4'] = f'{k1_SI}'
    SheetSI['T5'] = f'{k2_SI}'
    SheetSI['T6'] = f'{k3_SI}'
    SheetUS['S4'] = 'k1'
    SheetUS['S5'] = 'k2'
    SheetUS['S6'] = 'k3'
    SheetUS['S7'] = 'R2'
    SheetSI['S4'] = 'k1'
    SheetSI['S5'] = 'k2'
    SheetSI['S6'] = 'k3'
    SheetSI['S7'] = 'R2'
    SheetUS['S10'] = 'Please re-calibrate using the Excel Solver or fix the solver part of the code!'
    SheetSI['S10'] = 'Please re-calibrate using the Excel Solver or fix the solver part of the code!'

    # Overwrote the workbook
    Workbook.save(ExcelFileName)
    print(f'* Results has been saved to an Excel sheet next to the input CSV file: {ExcelFileName}')

    # Return Nothing.
    return
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def ExportData2Excel_PD(InputFile, CycleInfo, SeqInfo, PDResults):
    """
    This function exports the data in a form of an excel file.
    :param InputFile: The path to the input CSV file.
    :return: Nothing.
    """
    # Create the path for writing the Excel file.
    ExcelFileName   = os.path.join(os.path.dirname(InputFile),
                                   os.path.splitext(os.path.basename(InputFile))[0] + '-PD-Output.xlsx')

    # And now, write the sequence information in another sheet within that excel file.
    SeqInfo_SI = SeqInfo.copy()
    SeqInfo_US = SeqInfo.copy()
    SeqInfo_US['Deviator Stress (psi)'] = SeqInfo_US['Deviator Stress (kPa)'] * 0.145038
    SeqInfo_US.drop('Deviator Stress (kPa)', axis=1, inplace=True)
    SeqInfo_US.drop('MR (MPa)', axis=1, inplace=True)
    SeqInfo_SI.drop('MR (psi)', axis=1, inplace=True)
    # First add some more rows to the SeqInfo dataframe.
    SeqInfo_SI['Air Pressure (kPa)'] = 102.201  # Use the value for Lansing, MI in Nov 15, 2023
    SeqInfo_US['Air Pressure (psi)'] = SeqInfo_SI['Air Pressure (kPa)'] * 0.145038
    try:
        Confine_AASHTOt307_psi = [15, PDResults['ConfineStrs']]
        if len(SeqInfo_US) <= len(Confine_AASHTOt307_psi):
            SeqInfo_US['Confine Pressure (psi)'] = Confine_AASHTOt307_psi[:len(SeqInfo_US)]
        else:
            SeqInfo_US['Confine Pressure (psi)'] = Confine_AASHTOt307_psi + \
                                                   [0 for i in range(len(SeqInfo_US) - len(Confine_AASHTOt307_psi))]
    except:
        SeqInfo_US['Confine Pressure (psi)'] = 0
    SeqInfo_SI['Confine Pressure (kPa)'] = SeqInfo_US['Confine Pressure (psi)'] / 0.145038
    SeqInfo_US['σ3 (psi)'] = 0
    SeqInfo_SI['σ3 (kPa)'] = 0
    SeqInfo_US['σ1 (psi)'] = 0
    SeqInfo_SI['σ1 (kPa)'] = 0
    SeqInfo_US['τoct (psi)'] = 0
    SeqInfo_SI['τoct (kPa)'] = 0
    SeqInfo_US['Bulk (psi)'] = 0
    SeqInfo_SI['Bulk (kPa)'] = 0
    SeqInfo_US['Bulk/Pa'] = 0
    SeqInfo_SI['Bulk/Pa'] = 0
    SeqInfo_US['τoct/Pa+1'] = 0
    SeqInfo_SI['τoct/Pa+1'] = 0
    SeqInfo_US['Predict MR (psi)'] = 0
    SeqInfo_SI['Predict MR (MPa)'] = 0
    SeqInfo_US['MR MAE (psi)'] = 0
    SeqInfo_SI['MR MAE (MPa)'] = 0

    # Create a new DataFrame for the plastic strains.
    PlasticStrn = pd.DataFrame({'Cycle_Num': PDResults['CycleNum'],
                                'PlasticStrn': PDResults['PlasticStrn']})

    # Now writing them in two different sheets, as well as one more sheet for cycle information details.
    with pd.ExcelWriter(ExcelFileName, engine='openpyxl') as writer:
        SeqInfo_US.to_excel(writer, sheet_name='Seq_Info_US', index=False)
        SeqInfo_SI.to_excel(writer, sheet_name='Seq_Info_SI', index=False)
        CycleInfo.to_excel(writer, sheet_name='Cycle_Info', index=False)
        PlasticStrn.to_excel(writer, sheet_name='Plastic_Strain', index=False)

    # Now, modifying the sequence information sheets and provide the formulas in the cells.
    Workbook = openpyxl.load_workbook(ExcelFileName)            # Read the excel file with OpenPyXL module.
    SheetUS  = Workbook['Seq_Info_US']                          # Read the US sheet.
    SheetSI  = Workbook['Seq_Info_SI']                          # Read the US sheet.

    # Add a formula to the Cells.
    for i in range(len(SeqInfo_US)):
        SheetUS[f'H{2 + i}'] = f'=G{2 + i}'                     # Fixing the σ3.
        SheetSI[f'H{2 + i}'] = f'=G{2 + i}'
        SheetUS[f'I{2 + i}'] = f'=H{2 + i} + E{2 + i}'          # Fixing the σ1.
        SheetSI[f'I{2 + i}'] = f'=H{2 + i} + E{2 + i}'
        SheetUS[f'J{2 + i}'] = f'=1/3*SQRT((I{2 + i}-H{2 + i})^2+(I{2 + i}-H{2 + i})^2)'    # Fixing τoct.
        SheetSI[f'J{2 + i}'] = f'=1/3*SQRT((I{2 + i}-H{2 + i})^2+(I{2 + i}-H{2 + i})^2)'
        SheetUS[f'K{2 + i}'] = f'=I{2 + i}+H{2 + i}*2'          # Fixing the bulk stress.
        SheetSI[f'K{2 + i}'] = f'=I{2 + i}+H{2 + i}*2'
        SheetUS[f'L{2 + i}'] = f'=K{2 + i} / F{2 + i}'          # Fixing the bulk/Pa ratio.
        SheetSI[f'L{2 + i}'] = f'=K{2 + i} / F{2 + i}'
        SheetUS[f'M{2 + i}'] = f'=J{2 + i} / F{2 + i} + 1'      # Fixing the τoct/Pa+1 ratio.
        SheetSI[f'M{2 + i}'] = f'=J{2 + i} / F{2 + i} + 1'
        SheetUS[f'N{2 + i}'] = f'=($T$4*F{2 + i}*L{2 + i}^$T$5*M{2 + i}^$T$6)'      # Fixing the predicted MR (psi).
        SheetSI[f'N{2 + i}'] = f'=($T$4*F{2 + i}*L{2 + i}^$T$5*M{2 + i}^$T$6)/1000' # MR in MPa
        SheetUS[f'O{2 + i}'] = f'=ABS(N{2 + i} - C{2 + i})'     # Fixing the MR MAE.
        SheetSI[f'O{2 + i}'] = f'=ABS(N{2 + i} - C{2 + i})'
        SheetUS['T7'] = f'=RSQ((N3:N17),(C3:C17))'              # Fixing the R-square values.
        SheetSI['T7'] = f'=RSQ((N3:N17),(C3:C17))'

    # Add the k values.
    k1_US, k2_US, k3_US, k1_SI, k2_SI, k3_SI = Calc_kvalues(SeqInfo_US.copy(), SeqInfo_SI.copy())
    SheetUS['T4'] = f'{k1_US}'
    SheetUS['T5'] = f'{k2_US}'
    SheetUS['T6'] = f'{k3_US}'
    SheetSI['T4'] = f'{k1_SI}'
    SheetSI['T5'] = f'{k2_SI}'
    SheetSI['T6'] = f'{k3_SI}'
    SheetUS['S4'] = 'k1'
    SheetUS['S5'] = 'k2'
    SheetUS['S6'] = 'k3'
    SheetUS['S7'] = 'R2'
    SheetSI['S4'] = 'k1'
    SheetSI['S5'] = 'k2'
    SheetSI['S6'] = 'k3'
    SheetSI['S7'] = 'R2'
    SheetUS['S10'] = 'Please re-calibrate using the Excel Solver or fix the solver part of the code!'
    SheetSI['S10'] = 'Please re-calibrate using the Excel Solver or fix the solver part of the code!'

    # Overwrote the workbook
    Workbook.save(ExcelFileName)
    print(f'* Results has been saved to an Excel sheet next to the input CSV file: {ExcelFileName}')

    # Return Nothing.
    return
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def Calc_kvalues(SeqInfo_US, SeqInfo_SI):
    """
    This function estimates the k-values for the SI and US models.
    :param SeqInfo_US:
    :param SeqInfo_SI:
    :return:
    """
    # Calculating the Bulk, octahedral, and their ratios.
    SeqInfo_US['σ3 (psi)']          = SeqInfo_US['Confine Pressure (psi)']
    SeqInfo_SI['σ3 (kPa)']          = SeqInfo_SI['Confine Pressure (kPa)']
    SeqInfo_US['σ1 (psi)']          = SeqInfo_US['Confine Pressure (psi)'] + SeqInfo_US['Deviator Stress (psi)']
    SeqInfo_SI['σ1 (kPa)']          = SeqInfo_SI['Confine Pressure (kPa)'] + SeqInfo_SI['Deviator Stress (kPa)']
    SeqInfo_US['τoct (psi)']        = (1 / 3) * np.sqrt(2 * ((SeqInfo_US['σ1 (psi)'] - SeqInfo_US['σ3 (psi)']) ** 2))
    SeqInfo_SI['τoct (kPa)']        = (1 / 3) * np.sqrt(2 * ((SeqInfo_SI['σ1 (kPa)'] - SeqInfo_SI['σ3 (kPa)']) ** 2))
    SeqInfo_US['Bulk (psi)']        = SeqInfo_US['σ1 (psi)'] + 2 * SeqInfo_US['σ3 (psi)']
    SeqInfo_SI['Bulk (kPa)']        = SeqInfo_SI['σ1 (kPa)'] + 2 * SeqInfo_SI['σ3 (kPa)']
    SeqInfo_US['Bulk/Pa']           = SeqInfo_US['Bulk (psi)'] / SeqInfo_US['Air Pressure (psi)']
    SeqInfo_SI['Bulk/Pa']           = SeqInfo_SI['Bulk (kPa)'] / SeqInfo_SI['Air Pressure (kPa)']
    SeqInfo_US['τoct/Pa+1']         = (SeqInfo_US['τoct (psi)'] / SeqInfo_US['Air Pressure (psi)']) + 1
    SeqInfo_SI['τoct/Pa+1']         = (SeqInfo_SI['τoct (kPa)'] / SeqInfo_SI['Air Pressure (kPa)']) + 1

    # Define a objective function value to be used in optimization.
    def Calc_MR_MAR(kval, BulkPa, OctPa, Pa, MR, isMPa=False):
        Prediction = kval[0] * Pa * (BulkPa ** kval[1]) * (OctPa ** kval[2])
        if isMPa:
            Prediction /= 1e3
        return np.abs(Prediction - MR).mean()

    # Call the optimization functions.
    Options = {
        'maxiter': 5000,  # Maximum number of iterations
        'maxfev': 50000,  # Maximum number of function evaluations
        'fatol': 1,
        'disp': False      # Set to True to print convergence messages
    }
    result_US = minimize(lambda k: Calc_MR_MAR(k,
                                               SeqInfo_US['Bulk/Pa'].to_numpy()[1:16],
                                               SeqInfo_US['τoct/Pa+1'].to_numpy()[1:16],
                                               SeqInfo_US['Air Pressure (psi)'].to_numpy()[1:16],
                                               SeqInfo_US['MR (psi)'].to_numpy()[1:16], False),
                         np.array([1e5, 0.5, -0.20]),
                         bounds=[(1e3, 5e7), (0.1, 1.5), (-2.0, 2.0)], options=Options)
    result_SI = minimize(lambda k: Calc_MR_MAR(k,
                                               SeqInfo_SI['Bulk/Pa'].to_numpy()[1:16],
                                               SeqInfo_SI['τoct/Pa+1'].to_numpy()[1:16],
                                               SeqInfo_SI['Air Pressure (kPa)'].to_numpy()[1:16],
                                               SeqInfo_SI['MR (MPa)'].to_numpy()[1:16], True),
                         np.array([1e5, 0.5, -0.20]),
                         bounds=[(1e3, 5e7), (0.1, 1.5), (-2.0, 2.0)], options=Options)
    # The optimal parameters
    k1_US, k2_US, k3_US = result_US.x
    k1_SI, k2_SI, k3_SI = result_SI.x

    # Return the values.
    return k1_US, k2_US, k3_US, k1_SI, k2_SI, k3_SI
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


if __name__ == '__main__':
    main()
