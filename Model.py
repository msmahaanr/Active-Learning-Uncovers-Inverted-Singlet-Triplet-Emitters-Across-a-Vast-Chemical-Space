import os
import re
import sys
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = Path(os.environ.get("BASE_DIR", "."))
FILE_NAME = os.environ.get("FILE_NAME", "input.xlsx")
INITIAL_DIVERSE_SELECTION = int(os.environ.get("INITIAL_DIVERSE_SELECTION", 1000))
N_MOLECULES_PER_CYCLE = int(os.environ.get("N_MOLECULES_PER_CYCLE", 350))
DISCOVERY_RATE_THRESHOLD = float(os.environ.get("DISCOVERY_RATE_THRESHOLD", 20))
N_CPUS = int(os.environ.get("N_CPUS", 128))


# ============================================================================
# UTILITIES
# ============================================================================
def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048))
    except Exception:
        pass
    return None


def mmff_to_xyz(smiles, mol_id, output_dir):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        if ff:
            ff.Minimize(maxIts=200)
        (output_dir / f"{mol_id}.xyz").write_text(Chem.MolToXYZBlock(mol))
        return True
    except Exception:
        return False


# ============================================================================
# CYCLE 0: INITIAL DIVERSE SELECTION VIA K-MEANS
# ============================================================================
def initial_diverse_selection(base_dir, file_name, n_molecules):
    input_file = base_dir / "cycle_0" / file_name
    df = pd.read_excel(input_file)
    df["mol_id"] = range(1, len(df) + 1)

    df["fp"] = df["SMILES"].apply(get_fingerprint)
    df = df.dropna(subset=["fp"])

    X = np.array(df["fp"].tolist())
    kmeans = KMeans(n_clusters=n_molecules, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    selected_indices = []
    for i in range(n_molecules):
        mask = labels == i
        if np.any(mask):
            cluster_points = X[mask]
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            idx_in_cluster = np.argmin(dists)
            global_idx = np.where(mask)[0][idx_in_cluster]
            selected_indices.append(global_idx)

    selected_df = df.iloc[selected_indices].copy()
    selected_df[["mol_id", "SMILES"]].to_csv(
        base_dir / "cycle_0" / "selected_molecules.csv", index=False
    )
    df.to_pickle(base_dir / "full_database.pkl")


# ============================================================================
# STEP 1: xTB GEOMETRY OPTIMIZATION
# ============================================================================
def run_xtb(xyz_file, mol_id, xtb_dir):
    work_dir = xtb_dir / f"xtb_work_{mol_id}"
    work_dir.mkdir(exist_ok=True)
    shutil.copy(xyz_file, work_dir / "input.xyz")
    cmd = [
        "xtb", "input.xyz", "--opt", "normal", "--gfn", "2",
        "--chrg", "0", "--uhf", "0", "--parallel", str(N_CPUS),
    ]
    try:
        result = subprocess.run(
            cmd, cwd=work_dir, capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0 and (work_dir / "xtbopt.xyz").exists():
            shutil.copy(work_dir / "xtbopt.xyz", xtb_dir / f"{mol_id}_opt.xyz")
            return True
    except Exception:
        pass
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    return False


def step_xtb_optimization(cycle_dir, smiles_file):
    xtb_dir = cycle_dir / "xtb_opt"
    xtb_dir.mkdir(exist_ok=True)
    df = pd.read_csv(smiles_file)

    for _, row in df.iterrows():
        if mmff_to_xyz(row["SMILES"], row["mol_id"], xtb_dir):
            xyz_file = xtb_dir / f"{row['mol_id']}.xyz"
            run_xtb(xyz_file, row["mol_id"], xtb_dir)


# ============================================================================
# STEP 2: CREATE QBICS GROUND-STATE INPUTS
# ============================================================================
def step_create_qbics_gs_inputs(cycle_dir):
    xtb_dir = cycle_dir / "xtb_opt"
    qbics_dir = cycle_dir / "qbics_gs"
    qbics_dir.mkdir(exist_ok=True)

    for xyz_file in sorted(xtb_dir.glob("*_opt.xyz")):
        mol_id = xyz_file.stem.replace("_opt", "")
        lines = xyz_file.read_text().splitlines()[2:]
        coords = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                atom, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
                coords.append(f"    {atom:<2} {x:>15.8f} {y:>15.8f} {z:>15.8f}")
        coordinates = "\n".join(coords)

        inp = f"""basis
    def2-svp
end

scf
    charge  0
    spin2p1 1
    type    R
    max_it  250
end

mol
{coordinates}
end

task
    energy b3lyp
end
"""
        (qbics_dir / f"{mol_id}_gs.inp").write_text(inp)


# ============================================================================
# STEP 3: RUN QBICS GROUND-STATE CALCULATIONS
# ============================================================================
def step_run_qbics_gs(cycle_dir):
    qbics_dir = cycle_dir / "qbics_gs"
    for inp_file in sorted(qbics_dir.glob("*_gs.inp")):
        base = inp_file.stem
        subdir = qbics_dir / base
        subdir.mkdir(exist_ok=True)
        shutil.move(str(inp_file), str(subdir / inp_file.name))
        out_file = subdir / f"{base}.out"
        with open(out_file, "w") as fh:
            subprocess.run(
                ["qbics-linux-cpu", inp_file.name, "-n", str(N_CPUS)],
                cwd=subdir, stdout=fh, stderr=subprocess.STDOUT,
            )


# ============================================================================
# STEP 4: CREATE QBICS EXCITED-STATE (S1, T1) INPUTS
# ============================================================================
def step_create_excited_inputs(cycle_dir):
    gs_dir = cycle_dir / "qbics_gs"
    exc_dir = cycle_dir / "qbics_excited"
    exc_dir.mkdir(exist_ok=True)

    gs_folders = [f for f in gs_dir.iterdir() if f.is_dir() and f.name.endswith("_gs")]

    for gs in gs_folders:
        num = gs.name.split("_")[0]
        out_f = gs / f"{num}_gs.out"
        mwfn_f = gs / f"{num}_gs.mwfn"
        inp_f = gs / f"{num}_gs.inp"
        if not all(p.exists() for p in (out_f, mwfn_f, inp_f)):
            continue

        content = out_f.read_text()
        if "Normal termination at" not in content:
            continue

        inp = inp_f.read_text()
        m0 = inp.find("mol")
        m1 = inp.find("end", m0)
        coord_lines = [
            l for l in inp[m0:m1].splitlines()[1:] if l.strip() and "end" not in l
        ]
        coordinates = "\n".join(coord_lines)

        e_match = re.search(r"# of electrons:\s*(\d+)", content)
        a_match = re.search(r"# of alpha electrons:\s*(\d+)", content)
        b_match = re.search(r"# of basis functions:\s*(\d+)", content)
        if not (e_match and a_match and b_match):
            continue

        total_e = int(e_match.group(1))
        alpha_e = int(a_match.group(1))
        basis_f = int(b_match.group(1))

        s1_orb1 = f"{total_e} 1 1-{basis_f-1} : 1-{alpha_e-1} {alpha_e+1}-{basis_f}"
        s1_orb2 = f"0 1 {basis_f} : {alpha_e}"

        s1_content = f"""basis
    def2-svp
end

scf
    charge  0
    spin2p1 1
    type    U
    no_scf  tso
end

scfguess
    type mwfn
    file {mwfn_f.absolute()}
    orb  {s1_orb1}
    orb  {s1_orb2}
end

mol
{coordinates}
end

task
    energy b3lyp
end
"""

        t1_content = f"""basis
    def2-svp
end

scf
    charge  0
    spin2p1 3
    max_it  200
end

scfguess
    type mwfn
    file {mwfn_f.absolute()}
end

mol
{coordinates}
end

task
    energy b3lyp
end
"""

        mol_dir = exc_dir / f"{num}_S1"
        mol_dir.mkdir(exist_ok=True)
        (mol_dir / f"{num}_S1.inp").write_text(s1_content)
        (mol_dir / f"{num}_T1.inp").write_text(t1_content)


# ============================================================================
# STEP 5: RUN QBICS EXCITED-STATE CALCULATIONS
# ============================================================================
def step_run_qbics_excited(cycle_dir):
    exc_dir = cycle_dir / "qbics_excited"
    for s1_dir in sorted(p for p in exc_dir.glob("*_S1") if p.is_dir()):
        num = s1_dir.name.split("_")[0]
        for label in ("S1", "T1"):
            inp = s1_dir / f"{num}_{label}.inp"
            if not inp.exists():
                continue
            with open(s1_dir / f"{num}_{label}.out", "w") as fh:
                subprocess.run(
                    ["qbics-linux-cpu", inp.name, "-n", str(N_CPUS)],
                    cwd=s1_dir, stdout=fh, stderr=subprocess.STDOUT,
                )


# ============================================================================
# STEP 6: EXTRACT S1/T1 ENERGIES AND COMPUTE GAPS
# ============================================================================
def step_extract_results(cycle_dir):
    exc_dir = cycle_dir / "qbics_excited"
    results = []
    for s1_dir in sorted(exc_dir.glob("*_S1")):
        if not s1_dir.is_dir():
            continue
        num = s1_dir.name.split("_")[0]

        def extract(label):
            f = s1_dir / f"{num}_{label}.out"
            if not f.exists():
                return None
            m = re.search(r"E\(Current\)-E\(Ref\):\s*([-\d\.]+)\s*eV", f.read_text())
            return float(m.group(1)) if m else None

        s1 = extract("S1")
        t1 = extract("T1")
        if s1 is not None and t1 is not None:
            results.append({
                "mol_id": int(num),
                "S1": s1,
                "T1": t1,
                "gap": s1 - t1,
                "is_negative": (s1 - t1) < 0,
            })

    df = pd.DataFrame(results)
    res_dir = cycle_dir / "results"
    res_dir.mkdir(exist_ok=True)
    df.to_csv(res_dir / "cycle_results.csv", index=False)
    return df


# ============================================================================
# STEP 7: XGBOOST MODELING AND NEXT-BATCH SELECTION
# ============================================================================
def xgb_uncertainty(model, X):
    booster = model.get_booster()
    n_trees = len(booster.get_dump(with_stats=True))
    dmat = xgb.DMatrix(X)
    preds = np.array([
        booster.predict(dmat, iteration_range=(i, i + 1)) for i in range(n_trees)
    ])
    return np.std(preds, axis=0)


def train_classifier(X, y):
    n_neg = int(y.sum())
    n_pos = len(y) - n_neg
    spw = n_pos / n_neg if n_neg else 1.0
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        reg_lambda=1.0, reg_alpha=0.1, random_state=42,
        n_jobs=-1, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, y)
    return model


def attach_labels(df_all, df_calc):
    calc_ids = set(df_calc["mol_id"])
    df_all["calculated"] = df_all["mol_id"].isin(calc_ids)
    df_all["is_negative"] = False
    for _, row in df_calc.iterrows():
        df_all.loc[df_all["mol_id"] == row["mol_id"], "is_negative"] = row["is_negative"]
    return df_all


def ml_step(base_dir, cycle_num):
    df_all = pd.read_pickle(base_dir / "full_database.pkl")

    # Collect results from every cycle
    all_results, discovery_rates = [], []
    for c in range(cycle_num + 1):
        rf = base_dir / f"cycle_{c}" / "results" / "cycle_results.csv"
        if rf.exists():
            rdf = pd.read_csv(rf)
            all_results.append(rdf)
            if c > 0:
                n_neg = int(rdf["is_negative"].sum())
                n_tot = len(rdf)
                discovery_rates.append((n_neg / n_tot) * 100 if n_tot else 0)

    if not all_results:
        raise RuntimeError("No cycle results found")

    df_calc = pd.concat(all_results, ignore_index=True)
    df_calc = df_calc.drop_duplicates(subset=["mol_id"], keep="last")

    # Stopping criterion: last 3 cycles below threshold
    if len(discovery_rates) >= 3:
        if all(r < DISCOVERY_RATE_THRESHOLD for r in discovery_rates[-3:]):
            return "STOP"

    # Train classifier on all calculated molecules
    df_all = attach_labels(df_all, df_calc)
    X = np.array(df_all[df_all["calculated"]]["fp"].tolist())
    y = df_all[df_all["calculated"]]["is_negative"].values
    model = train_classifier(X, y)

    # Predict on uncalculated molecules
    uncalc = df_all[~df_all["calculated"]]
    X_uncalc = np.array(uncalc["fp"].tolist())
    prob_neg = model.predict_proba(X_uncalc)[:, 1]

    # Uncertainty from tree variance
    unc = xgb_uncertainty(model, X_uncalc)
    unc_score = unc / (unc.max() + 1e-6)

    # Acquisition: balance exploitation and exploration
    acquisition = 0.7 * prob_neg + 0.3 * unc_score

    top_idx = np.argsort(acquisition)[-N_MOLECULES_PER_CYCLE:]
    top_mol = uncalc.iloc[top_idx].copy()

    next_cycle_dir = base_dir / f"cycle_{cycle_num + 1}"
    next_cycle_dir.mkdir(exist_ok=True)
    top_mol[["mol_id", "SMILES"]].to_csv(
        next_cycle_dir / "selected_molecules.csv", index=False
    )
    return "CONTINUE"


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    cycle0 = BASE_DIR / "cycle_0"
    cycle0.mkdir(exist_ok=True)
    initial_diverse_selection(BASE_DIR, FILE_NAME, INITIAL_DIVERSE_SELECTION)

    cycle_num = 0
    while True:
        cycle_dir = BASE_DIR / f"cycle_{cycle_num}"
        smiles_file = cycle_dir / "selected_molecules.csv"
        if not smiles_file.exists():
            raise FileNotFoundError(f"Missing {smiles_file}")

        for sub in ("xtb_opt", "qbics_gs", "qbics_excited", "results"):
            (cycle_dir / sub).mkdir(exist_ok=True)

        step_xtb_optimization(cycle_dir, smiles_file)
        step_create_qbics_gs_inputs(cycle_dir)
        step_run_qbics_gs(cycle_dir)
        step_create_excited_inputs(cycle_dir)
        step_run_qbics_excited(cycle_dir)
        step_extract_results(cycle_dir)

        status = ml_step(BASE_DIR, cycle_num)
        if status == "STOP":
            return 0

        cycle_num += 1


if __name__ == "__main__":
    sys.exit(main())