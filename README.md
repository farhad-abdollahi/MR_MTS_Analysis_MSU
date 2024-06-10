# MR_MTS_Analysis_MSU

Author: Farhad Abdollahi (abdolla4@msu.edu) at Michigan State University.

Date: 06/10/2024

___

This repository includes codes for analyzing the results of resilient modulus and permanent deformation tests on unbound 
materials using the MTS machine. The code is specifically developed for the MTS machine in Michigan State University. 
The code can analyze both `csv` or `txt` format outputs from the MTS device. 

### Get Started

First, you need to have a Python 3 and proper libraries installed on your computer. For this purpose, you can use 
`conda` to create an environment for this project using the following steps (it is noted that you need to have anaconda
installed):  

1. Download the `environment.yml` file from the repository.
2. Open a terminal/CMD or Anaconda Prompt.
3. Navigate to the directory containing the `environment.yml` file.
4. Run the following command:
    ```bash
    conda env create -f environment.yml
    ```

### Run the code

After setting up the environment, you can activate it and run the code using the following command:

```bash
conda activate MR_MTS
python3 MR_MTS_Analysis.py
```

By runnig the code, a file selection window will pop-up and you can select your test result file in either `csv` or 
`txt` formats, as the MTS device produces. The code will analyze all loading cycles and loading sequences during the 
test. It will then ask for diameter and length of the sample to calculate the stress/strain and resilient modulus and 
return the results as an `excel` file in the same directory as the test results. In this regard, the following points 
are notable:

* The assumption of haversine loading followed by a rest time in each loading cycle is made.
* For MR tests, the confining cell pressure is assumed based on the AASHTO standard. You can modify the values in the
result `excel` file.
