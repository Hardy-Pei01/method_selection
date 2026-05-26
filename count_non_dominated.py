import numpy as np
import pandas as pd


def is_nondominated(objectives: np.ndarray) -> np.ndarray:

    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue

        diff = objectives - objectives[i]  # shape (n, n_obj)
        all_ge = np.all(diff >= 0, axis=1)  # every objective >= i
        any_gt = np.any(diff > 0, axis=1)  # at least one strictly >
        dominators = all_ge & any_gt
        dominators[i] = False  # a solution can't dominate itself
        if np.any(dominators):
            dominated[i] = True

    return ~dominated


def process_file(path: str) -> None:
    df = pd.read_csv(path)

    # Identify objective columns: all columns whose names start with 'r'
    # followed by digits (r0, r1, r2, …).
    obj_cols = [c for c in df.columns if c.startswith("r") and c[1:].isdigit()]

    if not obj_cols:
        print(f"[{path}] No objective columns found (expected r0, r1, …). Skipping.")
        return

    objectives = df[obj_cols].to_numpy(dtype=float)
    mask = is_nondominated(objectives)

    n_total = len(df)
    n_nd = int(mask.sum())

    print(f"File : {path}")
    print(f"  Objectives          : {obj_cols}")
    print(f"  Total solutions     : {n_total}")
    print(f"  Non-dominated (ND)  : {n_nd}")
    print(f"  Dominated           : {n_total - n_nd}")
    print(f"  ND ratio            : {n_nd / n_total:.1%}")
    print()


def main(path):

    try:
        process_file(path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
    except Exception as exc:
        print(f"[ERROR] Could not process {path}: {exc}")


if __name__ == "__main__":
    main("trees/depth9_dim2.csv")