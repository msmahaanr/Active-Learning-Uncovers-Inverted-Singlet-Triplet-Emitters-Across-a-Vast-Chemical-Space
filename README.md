Repository Structure

Folder 1. Optimization of Design Strategy
   Strategy-I
   Strategy-II
   Strategy-III

Each has:

- Designed Molecules SMILES
- TSO-DFT@B3LYP-def2-SVP_GFN2-xTB – Excitation properties calculated using TSO-DFT at the B3LYP/def2-SVP level on GFN2-xTB-optimized geometries
- TSO-DFT@B3LYP-def2-SVP_B3LYP-def2-SVP – Refined excitation properties calculated using TSO-DFT at the B3LYP/def2-SVP level on B3LYP/def2-SVP reoptimized geometries for molecules exhibiting negative ΔEST
- XYZ@B3LYP-def2-SVP – Ground-state and T1 excited-state Cartesian coordinates (XYZ format) from B3LYP/def2-SVP optimizations

Folder 2. Optimization of Active Learning
   i. Random Forest Model Data
   ii. XGBoost Model Data

Folder 3. XGBoost

Data for remaining four hydrocarbon families:

- Designed Molecules SMILES
- TSO-DFT@B3LYP-def2-SVP_GFN2-xTB – Excitation properties calculated using TSO-DFT at the B3LYP/def2-SVP level on GFN2-xTB-optimized geometries
- TSO-DFT@B3LYP-def2-SVP_B3LYP-def2-SVP – Refined excitation properties calculated using TSO-DFT at the B3LYP/def2-SVP level on B3LYP/def2-SVP reoptimized geometries for molecules exhibiting negative ΔEST
- XYZ@B3LYP-def2-SVP – Ground-state and T1 excited-state Cartesian coordinates (XYZ format) from B3LYP/def2-SVP optimizations
