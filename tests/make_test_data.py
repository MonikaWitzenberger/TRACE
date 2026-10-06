"""Create the small, made-up test data in tests/data (run once; the files are committed).

    python tests/make_test_data.py

Everything is random (fixed seed) and unrelated to any real experiment:

* ``transcripts.fa``          original sequences of four made-up transcripts
* ``ai_treated.csv`` /
  ``ai_control.csv``          count tables aligned to the original sequences
                              (A-to-I, C-to-U, m1A, ac4C, pseudouridine signals)
* ``m5c_treated.csv`` /
  ``m5c_control.csv``         count tables aligned to the C-to-T converted
                              sequences (bisulfite-like, for m5C)
* ``snps.csv``                a few positions to exclude
"""
from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).parent / "data"
SEED = 7
LENGTHS = {"tx_alpha": 150, "tx_beta": 120, "tx_gamma_2": 90, "spikeIn": 60}  # one name with "_"
BASES = "ACGT"


def sequences(rng: random.Random) -> dict[str, str]:
    seqs = {}
    for name, n in LENGTHS.items():
        s = [rng.choice(BASES) for _ in range(n)]
        for i in range(5, n - 3, 25):  # a few CCG motifs (ac4C_CCG) and CC (m5C_CC)
            s[i:i + 3] = "CCG"
        seqs[name] = "".join(s)
    return seqs


def counts(rng: random.Random, aligned: str, original: str, treated: bool, converted: bool) -> dict:
    """Read counts at one position; treated samples carry more 'modification' signal."""
    cov = rng.choice([0, 5, 15, 19, 20, 21, 40, 80, 150, 300])
    c = dict.fromkeys("ACGT-", 0)
    for _ in range(cov):
        r = rng.random()
        if converted and original == "C":
            # bisulfite: most C read as T, methylated ones stay C
            c["C" if r < (0.4 if treated else 0.05) else "T"] += 1
        elif r < (0.15 if treated else 0.02):
            c[{"A": "G", "C": "T", "G": "A", "T": "-"}[aligned]] += 1  # signal
        elif r < 0.17:
            c[rng.choice("ACGT")] += 1  # noise
        else:
            c[aligned] += 1
    return {"cov": cov, **c}


def table(rng: random.Random, seqs: dict[str, str], treated: bool, converted: bool) -> list[str]:
    rows = ["chr,gencoor,cov,refSeq,A,C,G,T,-"]
    for name, seq in seqs.items():
        if name == "tx_gamma_2" and not treated:
            continue  # transcript only present in the treated sample
        for pos, orig in enumerate(seq, start=1):
            if not treated and name == "spikeIn" and pos % 4 == 0:
                continue  # spike-in positions missing in the control
            if rng.random() < 0.03:
                continue  # a few positions missing
            aligned = "T" if converted and orig == "C" else orig
            c = counts(rng, aligned, orig, treated, converted)
            rows.append(f"{name},{pos},{c['cov']},{aligned},{c['A']},{c['C']},{c['G']},{c['T']},{c['-']}")
    return rows


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(exist_ok=True)
    seqs = sequences(rng)
    with open(OUT / "transcripts.fa", "w", newline="\n") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name} made-up test transcript\n")
            fh.writelines(seq[i:i + 60] + "\n" for i in range(0, len(seq), 60))
    for prefix, converted in (("ai", False), ("m5c", True)):
        for sample, treated in (("treated", True), ("control", False)):
            (OUT / f"{prefix}_{sample}.csv").write_text("\n".join(table(rng, seqs, treated, converted)) + "\n")
    (OUT / "snps.csv").write_text("pos\ntx_alpha_10\ntx_alpha_33\ntx_beta_7\ntx_gamma_2_40\nother_tx_5\n")


if __name__ == "__main__":
    main()
