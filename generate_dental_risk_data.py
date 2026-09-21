"""
DentiScan - Synthetic Dental Risk Dataset
===========================================
Real sources to swap in later:
  - NHANES Oral Health Examination + Dietary + Demographics (CDC), linked by SEQN
    https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination
  - Hugging Face "Oral Health & Dental Disease" dataset (30k records, 40+ variables)
    https://huggingface.co/datasets/Ontlametse/Tsu_Data

Both of those need real cleaning/merging work before they're a single flat table.
This script generates a *synthetic* dataset with the same kind of columns
(demographics + lifestyle + diet -> caries risk) purely so you can build and test
the full pipeline right now. Swap load_real_data() in for this once you've
downloaded and cleaned the real files -- the training script doesn't care where
the rows came from as long as the column names match.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2000


def generate(n=N) -> pd.DataFrame:
    age = RNG.integers(5, 80, size=n)
    gender = RNG.choice(["M", "F"], size=n)
    brushing_freq_per_day = RNG.choice([0, 1, 2, 3], size=n, p=[0.05, 0.25, 0.55, 0.15])
    flosses = RNG.choice([0, 1], size=n, p=[0.7, 0.3])
    sugary_snacks_per_day = RNG.poisson(1.5, size=n)
    smokes = RNG.choice([0, 1], size=n, p=[0.85, 0.15])
    dental_visit_last_year = RNG.choice([0, 1], size=n, p=[0.4, 0.6])
    family_income_level = RNG.choice(["low", "mid", "high"], size=n, p=[0.3, 0.5, 0.2])

    # Underlying risk score drives the probability of caries -- this encodes
    # the same kind of relationships the literature reports (brushing, sugar,
    # smoking, access to care, age).
    income_penalty = np.select(
        [family_income_level == "low", family_income_level == "mid", family_income_level == "high"],
        [0.6, 0.2, 0.0],
    )
    risk_score = (
        0.35 * sugary_snacks_per_day
        - 0.5 * brushing_freq_per_day
        - 0.4 * flosses
        + 0.5 * smokes
        - 0.3 * dental_visit_last_year
        + income_penalty
        + 0.02 * age
        + RNG.normal(0, 1.0, size=n)  # noise
    )
    prob_caries = 1 / (1 + np.exp(-(risk_score - 1.5)))
    has_caries = RNG.binomial(1, prob_caries)

    df = pd.DataFrame({
        "age": age,
        "gender": gender,
        "brushing_freq_per_day": brushing_freq_per_day,
        "flosses": flosses,
        "sugary_snacks_per_day": sugary_snacks_per_day,
        "smokes": smokes,
        "dental_visit_last_year": dental_visit_last_year,
        "family_income_level": family_income_level,
        "has_caries": has_caries,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("dental_risk_data.csv", index=False)
    print(df.head(10).to_string(index=False))
    print(f"\nGenerated {len(df)} rows -> dental_risk_data.csv")
    print(f"Caries prevalence in sample: {df['has_caries'].mean():.2%}")
