

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
- Recommended: Use a virtual environment

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
