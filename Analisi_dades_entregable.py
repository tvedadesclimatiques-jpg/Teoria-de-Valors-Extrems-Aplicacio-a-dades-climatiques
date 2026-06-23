from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EBRE_DATA_PATH = Path("Dades/t_1880_2024_mx_mm.dat")
FABRA_DATA_PATH = Path("Dades/PPT_TX_TN_diari_1914-2024.txt")
EBRE_OUTPUT_DIR = Path("analisi_ebre_output_entregable")
FABRA_OUTPUT_DIR = Path("analisi_fabra_output_entregable")
MISSING_VALUES = [99, 99.9, 999.9, -99, -99.9, -999.9]


def carregar_dades_ebre(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, names=["any", "mes", "dia", "tmax", "tmin"], na_values=MISSING_VALUES, encoding="latin1", engine="python")
    df["data"] = pd.to_datetime({"year": df["any"], "month": df["mes"], "day": df["dia"]}, errors="coerce")
    df = df[["data", "any", "mes", "dia", "tmax", "tmin"]].copy()
    for columna in ["tmax", "tmin"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df.loc[df[columna].isin(MISSING_VALUES), columna] = pd.NA
        df.loc[df[columna].abs() >= 90, columna] = pd.NA
    return df.dropna(subset=["data"]).sort_values("data").copy()


def carregar_dades_fabra(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=10, names=["any", "mes", "dia", "ppt", "tmax", "tmin"])
    for columna in ["any", "mes", "dia", "ppt", "tmax", "tmin"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
    df = df.dropna(subset=["any", "mes", "dia"])
    df["data"] = pd.to_datetime({"year": df["any"].astype(int), "month": df["mes"].astype(int), "day": df["dia"].astype(int)}, errors="coerce")
    df = df[(df["any"] >= 1914) & (df["any"] <= 2024)][["data", "any", "mes", "dia", "tmax", "tmin"]].copy()
    for columna in ["tmax", "tmin"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        df.loc[df[columna].isin(MISSING_VALUES), columna] = pd.NA
        df.loc[df[columna].abs() >= 90, columna] = pd.NA
    return df.dropna(subset=["data"]).sort_values("data").copy()


def anomalies_desestacionalitzades(df: pd.DataFrame) -> pd.DataFrame:
    df_daily = df.groupby("data")[["tmax", "tmin"]].mean().asfreq("D").reset_index()
    t = (df_daily["data"] - df_daily["data"].min()).dt.days.to_numpy(dtype=float) / 365.25
    X = np.column_stack([np.ones(len(t)), t, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    for columna in ["tmax", "tmin"]:
        valid = df_daily[columna].notna().to_numpy()
        beta = np.linalg.lstsq(X[valid], df_daily.loc[valid, columna].to_numpy(dtype=float), rcond=None)[0]
        df_daily[f"{columna}_anom"] = df_daily[columna] - X @ beta
    return df_daily.set_index("data")


def plot_serie_diaria(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(df["data"], df["tmax"], lw=0.35, alpha=0.45, color="tab:red", label="Tmax diaria")
    ax.plot(df["data"], df["tmin"], lw=0.35, alpha=0.45, color="tab:blue", label="Tmin diaria")
    ax.set_ylabel("Temperatura (ºC)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plots_dir / "serie_diaria_tmax_tmin.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_histogrames_temperatura(df: pd.DataFrame, plots_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, columna, color, title in zip(axes, ["tmax", "tmin"], ["tab:red", "tab:blue"], [r"$T_{max}$", r"$T_{min}$"]):
        ax.hist(df[columna].dropna(), bins=80, density=True, color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Temperatura (ºC)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Densitat")
    fig.tight_layout()
    path = plots_dir / "histogrames_temperatura.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_missing_per_any(df: pd.DataFrame, plots_dir: Path) -> Path:
    missing_any = df.groupby("any")[["tmax", "tmin"]].apply(lambda grup: grup.isna().sum())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(missing_any.index, missing_any["tmax"], color="tab:red", alpha=0.7, label="Tmax")
    ax.bar(missing_any.index, missing_any["tmin"], bottom=missing_any["tmax"], color="tab:blue", alpha=0.7, label="Tmin")
    ax.set_xlabel("Any")
    ax.set_ylabel("Valors no disponibles")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plots_dir / "missing_per_any.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_acf_anomalies_diaries(df: pd.DataFrame, plots_dir: Path) -> Path:
    df_daily = anomalies_desestacionalitzades(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for columna, label, color in [("tmax_anom", "Tmax", "tab:red"), ("tmin_anom", "Tmin", "tab:blue")]:
        serie = pd.to_numeric(df_daily[columna], errors="coerce")
        n_valid = int(serie.notna().sum())
        lags = np.arange(1, min(60, n_valid - 2) + 1)
        acf = np.array([serie.autocorr(lag=int(lag)) for lag in lags], dtype=float)
        ax.plot(lags, acf, color=color, lw=1.5, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag (dies)")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plots_dir / "diagnostic_block_maxima_acf_anomalies_diaries.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_durada_clusters_excedencies(df: pd.DataFrame, plots_dir: Path) -> Path:
    df_daily = anomalies_desestacionalitzades(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    for columna, label, color, quantile, sentit in [("tmax_anom", "Tmax > q0,95", "tab:red", 0.95, "superior"), ("tmin_anom", "Tmin < q0,05", "tab:blue", 0.05, "inferior")]:
        threshold = df_daily[columna].quantile(quantile)
        mask = df_daily[columna] > threshold if sentit == "superior" else df_daily[columna] < threshold
        lengths = []
        current = 0
        for value in mask.fillna(False).astype(bool).to_numpy():
            if value:
                current += 1
            elif current:
                lengths.append(current)
                current = 0
        if current:
            lengths.append(current)
        bins = np.arange(1, max(lengths) + 2) - 0.5
        ax.hist(lengths, bins=bins, histtype="step", linewidth=1.8, color=color, label=label)
    ax.set_xlabel("Dies consecutius")
    ax.set_ylabel("Nombre de clusters")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path = plots_dir / "diagnostic_block_maxima_durada_clusters_excedencies.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


df_ebre = carregar_dades_ebre(EBRE_DATA_PATH)
df_fabra = carregar_dades_fabra(FABRA_DATA_PATH)

for nom_dataset, df, output_dir in [("ebre", df_ebre, EBRE_OUTPUT_DIR), ("fabra", df_fabra, FABRA_OUTPUT_DIR)]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        plot_serie_diaria(df, output_dir),
        plot_histogrames_temperatura(df, output_dir),
        plot_missing_per_any(df, output_dir),
        plot_acf_anomalies_diaries(df, output_dir),
        plot_durada_clusters_excedencies(df, output_dir),
    ]
