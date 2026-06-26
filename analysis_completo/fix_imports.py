import json, sys, pathlib

NOTEBOOK_PATH = pathlib.Path(r"c:/Users/obarquero/SynologyDrive/Drive/Documentos/Research/Biology_research/Tuberculosis_Helio_2026/analysis_completo/01_EDA_Estadistica.ipynb")

def fix_imports(notebook_path: pathlib.Path):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            new_src = []
            for line in src:
                if "import matplotlib.pysplot" in line:
                    new_src.append("import matplotlib.pyplot as plt, seaborn as sns\n")
                    changed = True
                else:
                    new_src.append(line)
            # Ensure scipy.stats import present
            if not any("from scipy import stats" in l or "import scipy.stats" in l for l in new_src):
                # Insert after numpy import
                for i, l in enumerate(new_src):
                    if l.startswith("import numpy"):
                        new_src.insert(i+1, "from scipy import stats\n")
                        changed = True
                        break
            cell["source"] = new_src
    if changed:
        backup = notebook_path.with_suffix('.ipynb.bak')
        notebook_path.replace(backup)
        with notebook_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=2)
        print(f"Fixed imports in {notebook_path}. Backup saved as {backup}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    fix_imports(NOTEBOOK_PATH)
