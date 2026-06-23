from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
from scipy.stats import genextreme
from scipy.optimize import minimize
import numpy as np
from statistics import NormalDist


PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "Dades"
FITXER_FABRA = "series_extrems_fabra.parquet"
FITXER_EBRE = "series_extrems_ebre.parquet"
FITXERS_PARQUET = [INPUT_DIR / FITXER_FABRA, INPUT_DIR / FITXER_EBRE]

OUTPUT_PLOTS_DIR = PROJECT_DIR / "plots" / "Fittings_max_hessian_fabra_ebre"
TEX_OUTPUT_DIR = PROJECT_DIR.parent / "Teoria_de_Valors_Extrems__Aplicació_a_dades_climàtiques"


FREQ_BLOCS = "YE"  # "D" dia, "ME" mensual, "W" setmanal, "YE" anual
MIN_OBSERVACIONS_BLOC = 330
NOMS_SERIES_LATEX = {"EBRE_TMAX": r"Ebre $T_{\max}$", "EBRE_TMIN_NEG": r"Ebre $T_{\min}$", "FABRA_TX": r"Fabra $T_{\max}$", "FABRA_TN_NEG": r"Fabra $T_{\min}$"}
ORDRE_SERIES = ["EBRE_TMAX", "EBRE_TMIN_NEG", "FABRA_TX", "FABRA_TN_NEG"]
SERIES_MINIMA_NEGADA = {"EBRE_TMIN_NEG", "FABRA_TN_NEG"}
BINS_HISTOGRAMA = {"EBRE_TMAX": 25, "EBRE_TMIN_NEG": 25, "FABRA_TX": 22, "FABRA_TN_NEG": 22}


parser = argparse.ArgumentParser(description="Ajust GEV d'extrems per a Fabra i Ebre.")
parser.add_argument("--mu_linear", action="store_true", help="Calcula tambe el model no estacionari mu(t)=mu0+mu1*t i guarda la taula resum.")

args = parser.parse_args()


# Funcions matemàtiques i ajustos GEV

def GEV_cdf(x, xi, mu, sigma):
    x = np.asarray(x)
    
    term = 1 + xi * ((x - mu) / sigma)
    
    # inicialitzar amb nan
    F = np.full_like(x, np.nan, dtype=float)
    
    mask = term > 0
    
    if abs(xi) < 1e-6:
        z = (x - mu) / sigma
        F = np.exp(-np.exp(-z))
    else:
        F[mask] = np.exp(-(term[mask]) ** (-1 / xi))
    
    return F


def GEV_pdf(x, xi, mu, sigma):
    x = np.asarray(x, dtype=float)
    
    # inicialitzar amb 0 (fora del domini la densitat és 0)
    f = np.zeros_like(x)
    
    # evitar problemes numèrics
    if sigma <= 0 or not np.isfinite(sigma):
        return np.full_like(x, np.nan)
    
    term = 1 + xi * ((x - mu) / sigma)
    
    mask = term > 0
    
    # Cas Gumbel (xi va a 0)
    if abs(xi) < 1e-6:
        z = (x - mu) / sigma
        f = (1 / sigma) * np.exp(-(z + np.exp(-z)))
        return f
    
    # Cas general GEV
    t = term[mask]
    
    f[mask] = ((1 / sigma) * t ** (-1/xi - 1) * np.exp(-(t) ** (-1/xi)))
    
    return f


def GEV_ppf(p, xi, mu, sigma):
    p = np.asarray(p)

    if np.isclose(xi, 0):
        # cas Gumbel
        return mu - sigma * np.log(-np.log(p))
    else:
        return mu - (sigma / xi) * (1 - (-np.log(p))**(-xi))


def perd_retorn(T, xi, mu, sigma):
    T = np.asarray(T, dtype=float)

    p = 1 - 1 / T

    if np.isclose(xi, 0):
        # cas Gumbel
        return mu - sigma * np.log(-np.log(p))
    else:
        # cas GEV
        return mu + (sigma / xi) * ((-np.log(p)) ** (-xi) - 1)

def log_vers_GEV(params, x):
    xi, sigma, mu = params
    m = len(x)

    BIG = 1e20

    if sigma <= 0 or not np.isfinite(sigma):
        return BIG

    if not np.all(np.isfinite(params)):
        return BIG

    # Cas Gumbel (xi ≈ 0)
    if abs(xi) < 1e-6:
        z = (x - mu) / sigma
        return -(
            -m * np.log(sigma)
            - np.sum(z)
            - np.sum(np.exp(-z))
        )

    # Cas GEV general
    term = 1 + xi * ((x - mu) / sigma)

    if np.any(term <= 0) or not np.all(np.isfinite(term)):
        return BIG

    val = -(
        -m * np.log(sigma)
        - (1 + 1 / xi) * np.sum(np.log(term))
        - np.sum(term ** (-1 / xi))
    )

    return val if np.isfinite(val) else BIG

def log_vers_GEV_mu_linear(params, x, t):
    xi, sigma, mu0, mu1 = params
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    mu = mu0 + mu1 * t

    BIG = 1e20

    if sigma <= 0 or not np.isfinite(sigma):
        return BIG

    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        return BIG

    z = (x - mu) / sigma

    if abs(xi) < 1e-6:
        logpdf = -np.log(sigma) - z - np.exp(-z)
    else:
        term = 1 + xi * z
        if np.any(term <= 0) or not np.all(np.isfinite(term)):
            return BIG
        logpdf = (
            -np.log(sigma)
            - (1 + 1 / xi) * np.log(term)
            - term ** (-1 / xi)
        )

    val = -np.sum(logpdf)
    return val if np.isfinite(val) else BIG

def temps_centrats_anys(dates):
    dates = pd.to_datetime(dates)
    anys = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
    t0 = float(anys.mean())
    return (anys - t0).to_numpy(dtype=float), t0

def ajustar_gev_scipy(x):
    c_opt, mu_opt, sigma_opt = genextreme.fit(x, method="MLE")
    xi_opt = -c_opt
    return np.array([xi_opt, sigma_opt, mu_opt], dtype=float)

def ajustar_gev_mu_linear(x, t, params_estacionaris):
    xi_ini, sigma_ini, mu_ini = params_estacionaris
    initial_params = np.array([xi_ini, sigma_ini, mu_ini, 0.0], dtype=float)
    res = minimize(log_vers_GEV_mu_linear, initial_params, args=(x, t), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-9, "fatol": 1e-9})

    if not res.success or not np.isfinite(res.fun) or not np.all(np.isfinite(res.x)) or res.x[1] <= 0:
        raise RuntimeError("L'ajust GEV amb mu lineal no ha trobat parametres valids")

    return np.asarray(res.x, dtype=float)


# Intervals de confiança i bandes
def hessian_numerica(func, params, args=(), eps=1e-4, sigma_index=1):
    params = np.asarray(params, dtype=float)
    n = len(params)
    h = eps * np.maximum(np.abs(params), 1.0)

    # Evitar que el pas numeric faci sigma <= 0.
    if sigma_index is not None and sigma_index < n and params[sigma_index] - h[sigma_index] <= 0:
        h[sigma_index] = max(params[sigma_index] * 0.5, 1e-8)

    H = np.zeros((n, n), dtype=float)
    f0 = func(params, *args)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (func(params + ei, *args) - 2 * f0 + func(params - ei, *args)) / h[i]**2

        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = (
                func(params + ei + ej, *args)
                - func(params + ei - ej, *args)
                - func(params - ei + ej, *args)
                + func(params - ei - ej, *args)
            ) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]

    return H

def intervals_confianca_mle(func, params, args=(), alpha=0.05, sigma_index=1):
    H = hessian_numerica(func, params, args=args, sigma_index=sigma_index)

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)

    variances = np.diag(cov)
    se = np.full_like(params, np.nan, dtype=float)
    valid = np.isfinite(variances) & (variances >= 0)
    se[valid] = np.sqrt(variances[valid])

    z = NormalDist().inv_cdf(1 - alpha / 2)
    ci = np.column_stack((params - z * se, params + z * se))

    return ci, cov

def mostres_parametres_hessiana(params, cov, n_samples=2000, seed=12345):
    params = np.asarray(params, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if not np.all(np.isfinite(cov)):
        return np.empty((0, len(params)))

    # Fer la covariancia simetrica i semidefinida positiva si hi ha soroll numeric.
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(params, cov_psd, size=n_samples)
    valid = (samples[:, 1] > 0) & np.all(np.isfinite(samples), axis=1)

    return samples[valid]

def banda_parametrica_hessiana(funcio_valors, params, cov, alpha=0.05, n_samples=2000, seed=12345):
    samples = mostres_parametres_hessiana(params, cov, n_samples=n_samples, seed=seed)

    valors = []
    for theta_s in samples:
        y_s = funcio_valors(*theta_s)
        if np.all(np.isfinite(y_s)):
            valors.append(y_s)

    if len(valors) < 10:
        return None, None

    valors = np.asarray(valors)
    lower = np.nanpercentile(valors, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(valors, 100 * (1 - alpha / 2), axis=0)

    return lower, upper

def banda_delta_nivell_retorn(T, params, cov, alpha=0.05):
    params = np.asarray(params, dtype=float)
    T = np.asarray(T, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if not np.all(np.isfinite(cov)):
        return None, None

    eps = 1e-5
    n_params = len(params)
    grad = np.zeros((len(T), n_params), dtype=float)
    h = eps * np.maximum(np.abs(params), 1.0)
    if params[1] - h[1] <= 0:
        h[1] = max(params[1] * 0.5, 1e-8)

    for j in range(n_params):
        step = np.zeros(n_params)
        step[j] = h[j]
        z_plus = perd_retorn(T, (params + step)[0], (params + step)[2], (params + step)[1])
        z_minus = perd_retorn(T, (params - step)[0], (params - step)[2], (params - step)[1])
        grad[:, j] = (z_plus - z_minus) / (2 * h[j])

    cov = (cov + cov.T) / 2
    z_hat = perd_retorn(T, params[0], params[2], params[1])
    var = np.einsum("ij,jk,ik->i", grad, cov, grad)
    var = np.where(var >= 0, var, np.nan)
    se = np.sqrt(var)
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)

    return z_hat - z_alpha * se, z_hat + z_alpha * se


# Gràfics
def text_parametres_plot(key, params, ci):
    xi, sigma, mu = params
    if key in SERIES_MINIMA_NEGADA:
        mu = -mu
        mu_low = -ci[2, 1]
        mu_high = -ci[2, 0]
    else:
        mu_low = ci[2, 0]
        mu_high = ci[2, 1]

    return (
        rf"$\hat{{\xi}} = {xi:.3f}$ [{ci[0, 0]:.3f}, {ci[0, 1]:.3f}]" "\n"
        rf"$\hat{{\mu}} = {mu:.2f}$ [{mu_low:.2f}, {mu_high:.2f}]" "\n"
        rf"$\hat{{\sigma}} = {sigma:.2f}$ [{ci[1, 0]:.2f}, {ci[1, 1]:.2f}]"
    )

def histograma_pdf_plot(x, xi, mu, sigma, freq, key, station, plots_dir, params, ci, cov):
    x = np.asarray(x, dtype=float)
    x_plot = -x if key in SERIES_MINIMA_NEGADA else x
    x_vals = np.linspace(min(x_plot), max(x_plot), 200)
    x_model_vals = -x_vals if key in SERIES_MINIMA_NEGADA else x_vals

    y_vals = GEV_pdf(x_model_vals, xi, mu, sigma)
    y_low, y_high = banda_parametrica_hessiana(lambda xi_s, sigma_s, mu_s: GEV_pdf(x_model_vals, xi_s, mu_s, sigma_s), params, cov, alpha=0.05, n_samples=2000)

    ax = plt.gca()
    plt.xlabel(r"Temperatura ($^\circ$C)")
    plt.ylabel("Densitat")
    ax.set_title("")
    plt.hist(x_plot, bins=BINS_HISTOGRAMA[key], density=True, alpha=0.6, color='tab:orange', edgecolor='black', label='Dades')
    plt.fill_between(x_vals, y_low, y_high, color="green", alpha=0.20, label="Banda IC 95%")
    plt.plot(x_vals, y_vals, label='Dist. Inferida', linewidth=3, color='green')
    textstr = text_parametres_plot(key, params, ci)

    legend_loc = "upper left" if key in SERIES_MINIMA_NEGADA else "upper right"
    legend_anchor = (0.0, 0.78) if key in SERIES_MINIMA_NEGADA else (1.0, 0.78)
    main_legend = ax.legend(loc=legend_loc, fontsize="small")
    ax.add_artist(main_legend)
    param_handle = Line2D([], [], linestyle="none", label=textstr)
    ax.legend(handles=[param_handle], loc=legend_loc, bbox_to_anchor=legend_anchor, frameon=True, fontsize="small", handlelength=0, handletextpad=0, borderpad=0.6)
    plt.grid()
    plt.savefig(plots_dir / f"{freq}_{key}_{station['Nom'].iloc[0]}.png", dpi=300)
    plt.close()

def histograma_distribucio_plot(x, xi, mu, sigma, freq, key, station, plots_dir, params, ci, cov):
    x = np.asarray(x, dtype=float)
    x_plot = -x if key in SERIES_MINIMA_NEGADA else x
    x_vals = np.linspace(min(x_plot), max(x_plot), 200)
    x_model_vals = -x_vals if key in SERIES_MINIMA_NEGADA else x_vals

    y_vals = GEV_cdf(x_model_vals, xi, mu, sigma)
    y_low, y_high = banda_parametrica_hessiana(lambda xi_s, sigma_s, mu_s: GEV_cdf(x_model_vals, xi_s, mu_s, sigma_s), params, cov, alpha=0.05, n_samples=2000)
    if key in SERIES_MINIMA_NEGADA:
        y_vals = 1 - y_vals
        y_low, y_high = 1 - y_high, 1 - y_low

    ax = plt.gca()
    plt.xlabel(r"Temperatura ($^\circ$C)")
    plt.ylabel("Densitat acumulada")
    ax.set_title("")
    plt.hist(x_plot, bins=BINS_HISTOGRAMA[key], density=True, cumulative=True, alpha=0.6, color='tab:orange', edgecolor='black', label='Dades')
    plt.fill_between(x_vals, np.clip(y_low, 0, 1), np.clip(y_high, 0, 1), color="green", alpha=0.20, label="Banda IC 95%")
    plt.plot(x_vals, np.clip(y_vals, 0, 1), label='Dist. Inferida', linewidth=3, color='green')
    plt.ylim(-0.02, 1.02)
    textstr = text_parametres_plot(key, params, ci)

    legend_loc = "upper left" if key in SERIES_MINIMA_NEGADA else "lower right"
    legend_anchor = (0.0, 0.78) if key in SERIES_MINIMA_NEGADA else (1.0, 0.22)
    main_legend = ax.legend(loc=legend_loc, fontsize="small")
    ax.add_artist(main_legend)
    param_handle = Line2D([], [], linestyle="none", label=textstr)
    ax.legend(handles=[param_handle], loc=legend_loc, bbox_to_anchor=legend_anchor, frameon=True, fontsize="small", handlelength=0, handletextpad=0, borderpad=0.6)
    plt.grid()
    plt.savefig(plots_dir / f"{freq}_CDF_{key}_{station['Nom'].iloc[0]}.png", dpi=300)
    plt.close()

def probab_plot(x, xi, mu, sigma, freq, key, station, plots_dir, params, cov):
    if key in SERIES_MINIMA_NEGADA:
        x_plot_sorted = np.sort(-np.asarray(x, dtype=float))
        x_model = -x_plot_sorted
        hat_G = np.arange(1, len(x) + 1) / (len(x) + 1)
        G = 1 - GEV_cdf(x_model, xi, mu, sigma)
    else:
        x_sorted = np.sort(x)
        x_model = x_sorted
        hat_G = np.arange(1, len(x) + 1) / (len(x) + 1)
        G = GEV_cdf(x_model, xi, mu, sigma)

    G_low, G_high = banda_parametrica_hessiana(lambda xi_s, sigma_s, mu_s: GEV_cdf(x_model, xi_s, mu_s, sigma_s), params, cov)
    if key in SERIES_MINIMA_NEGADA:
        G_low, G_high = 1 - G_high, 1 - G_low
    plt.fill_between(hat_G, G_low, G_high, color="tab:orange", alpha=0.20, label="Banda IC 95%")
    plt.plot(hat_G, G, color="tab:orange", linewidth=2, label="Ajust GEV")
    plt.xlabel("Probabilitats observades")
    plt.ylabel("Probabilitats inferides")
    plt.plot([0,1], [0,1], "--", color="grey", alpha=0.65, linewidth=1, label="y = x")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{freq}_PP_{key}_{station['Nom'].iloc[0]}.png", dpi=300)
    plt.close()

def quantile_plot(x, xi, mu, sigma, freq, key, station, plots_dir, params, cov):
    x = np.asarray(x)
    x_plot_sorted = np.sort(-x if key in SERIES_MINIMA_NEGADA else x)
    n = len(x_plot_sorted)

    p = np.array([(i + 1) / (n + 1) for i in range(n)])
    if key in SERIES_MINIMA_NEGADA:
        p_model = 1 - p
        q_theoretical = -GEV_ppf(p_model, xi, mu, sigma)
    else:
        p_model = p
        q_theoretical = GEV_ppf(p_model, xi, mu, sigma)
    q_low, q_high = banda_parametrica_hessiana(lambda xi_s, sigma_s, mu_s: GEV_ppf(p_model, xi_s, mu_s, sigma_s), params, cov)
    if key in SERIES_MINIMA_NEGADA:
        q_low, q_high = -q_high, -q_low
    plt.fill_betweenx(x_plot_sorted, q_low, q_high, color="tab:orange", alpha=0.20, label="Banda IC 95%")

    plt.plot(q_theoretical, x_plot_sorted, color="tab:orange", linewidth=2, label="Ajust GEV")

    # línia y = x
    q_min = np.nanmin(q_theoretical)
    q_max = np.nanmax(q_theoretical)
    q_min = min(q_min, np.nanmin(q_low))
    q_max = max(q_max, np.nanmax(q_high))
    min_val = min(q_min, np.min(x_plot_sorted))
    max_val = max(q_max, np.max(x_plot_sorted))
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="grey", alpha=0.65, linewidth=1, label="y = x")

    plt.xlabel(r"Quantils inferits ($^\circ$C)")
    plt.ylabel(r"Quantils observats ($^\circ$C)")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / f"{freq}_Q_{key}_{station['Nom'].iloc[0]}.png", dpi=300)
    plt.close()

def perd_return_plot(x, xi, mu, sigma, freq, key, station, plots_dir, params, cov):
    T = np.linspace(1.01, 100, 400)
    nom = station["Nom"].iloc[0]

    z_p = perd_retorn(T, xi, mu, sigma)
    if key in SERIES_MINIMA_NEGADA:
        z_p = -z_p

    z_low, z_high = banda_delta_nivell_retorn(T, params, cov, alpha=0.05)
    if key in SERIES_MINIMA_NEGADA:
        z_low, z_high = -z_high, -z_low
    plt.fill_between(T, z_low, z_high, color="green", alpha=0.20, label="Banda IC 95% (delta)")

    plt.plot(T, z_p, color="green", linewidth=2, label="Inferit")
    ax = plt.gca()
    ax.set_title("")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    plt.xlabel("Període de retorn T (anys)")
    plt.ylabel(r"Nivell de retorn fred" if key in SERIES_MINIMA_NEGADA else r"Nivell de retorn $z_p$")
    plt.grid(True)

    

    plt.savefig(plots_dir / f"{freq}_PR_{key}_{nom}.png", dpi=300)

    plt.close()


# Taules
def parametres_estacionaris_taula(fila):
    if fila["codi"] in SERIES_MINIMA_NEGADA:
        mu = -fila["mu"]
        mu_low = -fila["mu_high"]
        mu_high = -fila["mu_low"]
    else:
        mu = fila["mu"]
        mu_low = fila["mu_low"]
        mu_high = fila["mu_high"]

    return {
        "xi": rf"${fila['xi']:.3f}\,({fila['xi_low']:.3f},\;{fila['xi_high']:.3f})$",
        "mu": rf"${mu:.2f}\,({mu_low:.2f},\;{mu_high:.2f})$",
        "sigma": rf"${fila['sigma']:.2f}\,({fila['sigma_low']:.2f},\;{fila['sigma_high']:.2f})$",
    }

def guardar_taula_estacionaria_tex(resultats):
    if not resultats:
        return

    TEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TEX_OUTPUT_DIR / "resum_gev_estacionaria_entregable.tex"
    linies = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Sèrie & $\hat{\xi}$ & $\hat{\mu}$ & $\hat{\sigma}$ \\",
        r"\hline",
    ]

    resultats_per_codi = {fila["codi"]: fila for fila in resultats}
    for codi in ORDRE_SERIES:
        if codi not in resultats_per_codi:
            continue
        fila = resultats_per_codi[codi]
        params = parametres_estacionaris_taula(fila)
        linies.append(" & ".join([NOMS_SERIES_LATEX[codi], params["xi"], params["mu"], params["sigma"]]) + r" \\")
        linies.append("")

    linies.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Estimacions dels paràmetres de les distribucions GVE per a les "
                r"sèries de temperatures extremes dels observatoris de l'Ebre i Fabra. "
                r"Els valors entre parèntesi corresponen als intervals de confiança del 95\%.}"
            ),
            r"\label{tab:gev_estacionaria}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(linies), encoding="utf-8")
    print(f"Taula GEV estacionaria guardada a: {path}")

def parametres_mu_linear_taula(fila):
    if fila["codi"] in SERIES_MINIMA_NEGADA:
        mu0 = -fila["mu0"]
        mu0_low = -fila["mu0_high"]
        mu0_high = -fila["mu0_low"]
        mu1 = -fila["mu1"]
        mu1_low = -fila["mu1_high"]
        mu1_high = -fila["mu1_low"]
    else:
        mu0 = fila["mu0"]
        mu0_low = fila["mu0_low"]
        mu0_high = fila["mu0_high"]
        mu1 = fila["mu1"]
        mu1_low = fila["mu1_low"]
        mu1_high = fila["mu1_high"]

    return [
        (r"$\xi$", fila["xi"], fila["xi_low"], fila["xi_high"]),
        (r"$\sigma$", fila["sigma"], fila["sigma_low"], fila["sigma_high"]),
        (r"$\mu_0$", mu0, mu0_low, mu0_high),
        (r"$\mu_1$ ($^\circ$C/any)", mu1, mu1_low, mu1_high),
    ]

def guardar_taula_mu_linear_tex(resultats):
    if not resultats:
        return

    TEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TEX_OUTPUT_DIR / "resum_mu_linear_entregable.tex"
    linies = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrlrrrr}",
        r"\hline",
        (
            r"Sèrie & Paràmetre & Estimació & IC 95\% & "
            r"AIC no est. & BIC no est. & AIC est. & BIC est. \\"
        ),
        r"\hline",
    ]

    resultats_per_codi = {fila["codi"]: fila for fila in resultats}
    for codi in ORDRE_SERIES:
        if codi not in resultats_per_codi:
            continue
        fila = resultats_per_codi[codi]
        parametres = parametres_mu_linear_taula(fila)
        for i, (parametre, estimacio, ci_low, ci_high) in enumerate(parametres):
            linies.append(
                " & ".join(
                    [
                        NOMS_SERIES_LATEX[codi] if i == 0 else "",
                        parametre,
                        f"{estimacio:.4f}",
                        f"[{ci_low:.4f}, {ci_high:.4f}]",
                        f"{fila['aic']:.2f}" if i == 0 else "",
                        f"{fila['bic']:.2f}" if i == 0 else "",
                        f"{fila['aic_estacionari']:.2f}" if i == 0 else "",
                        f"{fila['bic_estacionari']:.2f}" if i == 0 else "",
                    ]
                )
                + r" \\"
            )
        linies.append(r"\hline")

    linies.extend(
        [
            r"\end{tabular}",
            r"}",
            r"\caption{Estimacions dels paràmetres amb $\mu$ variant amb el temps. Comparació de AIC/BIC dels models estacionari i no estacionari ajustat sobre els mateixos blocs.}",
            r"\label{tab:mu_linear_gev}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(linies), encoding="utf-8")
    print(f"Taula mu_linear guardada a: {path}")

df = pd.concat([pd.read_parquet(fitxer) for fitxer in FITXERS_PARQUET], ignore_index=True)
df["DATA"] = pd.to_datetime(df["DATA"])
grouped = df.groupby("CODI")
resultats_estacionari = []
resultats_mu_linear = []
OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

for key, station in grouped:
    # Construcció dels blocs anuals d'extrems.
    blocs = station.groupby([pd.Grouper(key="DATA", freq=FREQ_BLOCS)])["original"]
    station_grouped = pd.DataFrame({"DATA": blocs.max().index, "original": blocs.max().values, "n_observacions": blocs.count().values})

    # Selecció dels blocs amb prou observacions.
    blocs_amb_prou_dades = station_grouped["n_observacions"] >= MIN_OBSERVACIONS_BLOC
    station_grouped = station_grouped[blocs_amb_prou_dades]

    # Extracció de la mostra final per a l'ajust.
    valid_x = station_grouped["original"].notna()
    x = station_grouped.loc[valid_x, "original"].values
    dates_x = station_grouped.loc[valid_x, "DATA"]
    t_anys, _ = temps_centrats_anys(dates_x)

    params_opt = ajustar_gev_scipy(x)
    xi_opt, sigma_opt, mu_opt = params_opt
    ci, cov = intervals_confianca_mle(log_vers_GEV, params_opt, args=(x,), alpha=0.05)

    resultats_estacionari.append(
        {
            "codi": key,
            "nom": station["Nom"].iloc[0],
            "freq": FREQ_BLOCS,
            "n": len(x),
            "xi": xi_opt,
            "xi_low": ci[0, 0],
            "xi_high": ci[0, 1],
            "mu": mu_opt,
            "mu_low": ci[2, 0],
            "mu_high": ci[2, 1],
            "sigma": sigma_opt,
            "sigma_low": ci[1, 0],
            "sigma_high": ci[1, 1],
        }
    )

    histograma_pdf_plot(x, xi_opt, mu_opt, sigma_opt, FREQ_BLOCS, key, station, OUTPUT_PLOTS_DIR, params_opt, ci, cov)
    histograma_distribucio_plot(x, xi_opt, mu_opt, sigma_opt, FREQ_BLOCS, key, station, OUTPUT_PLOTS_DIR, params_opt, ci, cov)
    probab_plot(x, xi_opt, mu_opt, sigma_opt, FREQ_BLOCS, key, station, OUTPUT_PLOTS_DIR, params_opt, cov)
    quantile_plot(x, xi_opt, mu_opt, sigma_opt, FREQ_BLOCS, key, station, OUTPUT_PLOTS_DIR, params_opt, cov)
    perd_return_plot(x, xi_opt, mu_opt, sigma_opt, FREQ_BLOCS, key, station, OUTPUT_PLOTS_DIR, params_opt, cov)

    if args.mu_linear:
        params_mu_linear = ajustar_gev_mu_linear(x, t_anys, params_opt)
        xi_ml, sigma_ml, mu0_ml, mu1_ml = params_mu_linear
        nll_opt = log_vers_GEV(params_opt, x)
        nll_mu_linear = log_vers_GEV_mu_linear(params_mu_linear, x, t_anys)
        ci_ml, _ = intervals_confianca_mle(log_vers_GEV_mu_linear, params_mu_linear, args=(x, t_anys), alpha=0.05, sigma_index=1)

        n_params = 4
        n_params_estacionari = 3
        aic = 2 * n_params + 2 * nll_mu_linear
        bic = n_params * np.log(len(x)) + 2 * nll_mu_linear
        aic_estacionari = 2 * n_params_estacionari + 2 * nll_opt
        bic_estacionari = n_params_estacionari * np.log(len(x)) + 2 * nll_opt

        resultats_mu_linear.append(
            {
                "codi": key,
                "nom": station["Nom"].iloc[0],
                "freq": FREQ_BLOCS,
                "n": len(x),
                "xi": xi_ml,
                "xi_low": ci_ml[0, 0],
                "xi_high": ci_ml[0, 1],
                "sigma": sigma_ml,
                "sigma_low": ci_ml[1, 0],
                "sigma_high": ci_ml[1, 1],
                "mu0": mu0_ml,
                "mu0_low": ci_ml[2, 0],
                "mu0_high": ci_ml[2, 1],
                "mu1": mu1_ml,
                "mu1_low": ci_ml[3, 0],
                "mu1_high": ci_ml[3, 1],
                "aic": aic,
                "bic": bic,
                "aic_estacionari": aic_estacionari,
                "bic_estacionari": bic_estacionari,
            }
        )

guardar_taula_estacionaria_tex(resultats_estacionari)

if args.mu_linear:
    guardar_taula_mu_linear_tex(resultats_mu_linear)
