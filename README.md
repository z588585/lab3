

#  HUAWEI ZHANG - ISE Coursework: Configuration Performance Tuning

## Project Overview

This is the Lab 3 coursework project by Huawei Zhang for the ISE course, focused on **Configuration Performance Tuning**.

I implemented and compared **6 intelligent tuning methods** along with a **random sampling baseline**, including:

- BestConfig  
- Linear BestConfig  
- FLASH  
- Linear FLASH  
- Fourier  
- Bayesian Optimization  
- Random Sampling (Baseline)

Additionally, the project includes a visualization module for easy analysis of the results.

>  For more details, please refer to the documents in the `doc/` folder.

---

## Project Structure

```
lab3\
│
├── 📂 datasets\                                  # Dataset folder
│
├── 📂 doc\                                       # Documentation folder
│   ├── 📜 CS_Mag_7_22.docx                       # Research paper
│   ├── 📜 lab3.pdf                               
│   ├── 📜 lab3cn.pdf                             
│   ├── 📜 manual.md                              # User manual
│   ├── 📜 replication.md                         # Replication guide
│   └── 📜 requirement.md                         # Requirements document
│
├── 📂 results\                                   # Results data
│   └── 📜 xxx_xxx.csv                            
│
├── 📂 results_img\                               # Result images
│   └── 📜 xxx_xxx.png                            
│
├── 📜 additional_code.py                         # Auxiliary code file
├── 📜 lab3_baseline.py                           # Baseline search algorithm
├── 📜 lab3_bayesian.py                           # Bayesian optimization algorithm
├── 📜 lab3_bestconfig.py                         # BestConfig search algorithm
├── 📜 lab3_bestconfig_linear.py                  # Linear BestConfig search algorithm
├── 📜 lab3_flash.py                              # FLASH search algorithm
├── 📜 lab3_flash_linear.py                       # Linear FLASH search algorithm
├── 📜 lab3_forier.py                             # Fourier search algorithm
├── 🐍 main.py                                    # Main program note: start here
├── 📓 visualization.ipynb                        # Visualization Jupyter notebook
├── 📜 README.md                                  # Project README file
└── 📜 requirements.txt                           # Project dependencies list

```

---

## Environment Setup

- Windows 10/11  
- VSCode  
- Python 3.10.11 (64-bit)  

### Install Dependencies

```bash
py -3.10 -m pip install -r requirements.txt
```

---

## Run Configuration Performance Tuning

> **Note:** All experiment results are already included in the repository. You don't need to rerun the tuning algorithms to view the visualizations.

To rerun the tuning process manually:

1. Open PowerShell or terminal.
2. Run:

```bash
py -3.10 main.py
```

- The program will complete tuning (excluding Bayesian) in about **1 hour**.
- **Bayesian Optimization** takes several hours — uncomment the corresponding line in `main.py` if you wish to run it.
- The default seed is `42`, which reproduces all results from the included research paper.

---

## Run Visualization

1. Open the project in VSCode.
2. Open and run `visualization.ipynb`.
3. View the results:
   - Visualizations will be saved in the `results_img/` folder.
   - Or directly view plots inside the notebook after execution.

---

## Documentation

You can find detailed project documents in the `doc/` folder:

- `lab3.pdf / lab3cn.pdf`: Project description (EN & CN)  
- `manual.md`: User manual  
- `replication.md`: Guide to replicate results  
- `requirement.md`: Environment requirements  
- `CS_Mag_7_22.docx`: Full research paper

---


## Run Tuning via Command Line or Python

You can run the full tuning process (excluding Bayesian optimization) from the command line:

```bash
py -3.10 main.py -na
```

This will apply the selected tuning algorithm to all configurable systems. Below is an example of the interactive prompts:

```
global_seed:  42
please set your dataset folder:
- datasets
please set your output folder:
- results\my_search_results
please set your budget:
- 50
please set your search method:
you can choose from: 1.bestconfig_fast 2.fourier 3.random 4.bestconfig 5.flash 6.flash_linear 7.bayesian
- 3
```

Once the search completes, the best configurations and performance results for each system will be shown in both the terminal and saved `.sav` files, for example:

```
System: 7z.csv (Seed: 6677)   Best Config:    [2, 0, 1, 0, 1, 70, 512, 2]   Best Performance: 4751.6
System: Apache.csv (Seed: 368)   Best Config:    [1, 1, 2, 1, 0, 1, 512, 0]   Best Performance: 32.099
...
```

> It is recommended to use `lab3_bestconfig_linear` or `lab3_flash_linear`, which often yield better performance under the same budget.

---

## Call Tuning via Python Code

You can also call the tuning process programmatically in your Python code:

```python
import main

bestconfig_search(
    datasets_folder="datasets",
    budget=50,
    output_folder="results/my_search_results",
    search_method="bestconfig_linear",  # or "flash_linear", etc.
    global_seed=42
)
```

This provides a flexible way to integrate tuning directly into your own workflow or scripts.

---
