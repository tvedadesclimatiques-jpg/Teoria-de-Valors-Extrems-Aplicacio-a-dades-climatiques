from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from statistics import NormalDist


PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "Dades"
FITXER_FABRA = "series_extrems_fabra.parquet"
FITXER_EBRE = "series_extrems_ebre.parquet"
FITXERS_PARQUET = [INPUT_DIR / FITXER_FABRA, INPUT_DIR / FITXER_EBRE]

OUTPUT_PLOTS_DIR = PROJECT_DIR / "plots" / "Fittings_GPD_entregable"
TEX_OUTPUT_DIR = PROJECT_DIR.parent / "Teoria_de_Valors_Extrems__Aplicació_a_dades_climàtiques"

THRESHOLD_QUANTILES = np.linspace(0.70, 0.995, 80)
WEEKLY_MAXIMA_FREQ = "W"
SERIES_MINIMA_NEGADA = {"FABRA_TN_NEG", "EBRE_TMIN_NEG"}
BINS_PDF_GPD = {"FABRA_TX": 18, "FABRA_TN_NEG": 16, "EBRE_TMAX": 22, "EBRE_TMIN_NEG": 18}
NOMS_SERIES_LATEX = {"EBRE_TMAX": r"Ebre $T_{\max}$", "EBRE_TMIN_NEG": r"Ebre $T_{\min}$", "FABRA_TX": r"Fabra $T_{\max}$", "FABRA_TN_NEG": r"Fabra $T_{\min}$"}
ORDRE_SERIES = ["EBRE_TMAX", "EBRE_TMIN_NEG", "FABRA_TX", "FABRA_TN_NEG"]

# Funcions matemàtiques i ajustos GPD
def log_vers_GPD(params, y):
    xi, beta = params
    n = len(y)
    BIG = 1e20

    if beta <= 0 or not np.all(np.isfinite(params)):
        return BIG

    if abs(xi) < 1e-6:
        val = n * np.log(beta) + np.sum(y) / beta
        return val if np.isfinite(val) else BIG

    term = 1 + xi * y / beta
    if np.any(term <= 0) or not np.all(np.isfinite(term)):
        return BIG

    val = n * np.log(beta) + (1 + 1 / xi) * np.sum(np.log(term))
    return val if np.isfinite(val) else BIG

def GPD_cdf(y, xi, beta):
    y = np.asarray(y, dtype=float)

    if beta <= 0:
        return np.full_like(y, np.nan)

    if abs(xi) < 1e-6:
        return 1 - np.exp(-y / beta)

    term = 1 + xi * y / beta
    F = np.full_like(y, np.nan)
    mask = term > 0
    F[mask] = 1 - term[mask] ** (-1 / xi)
    return F

def GPD_pdf(y, xi, beta):
    y = np.asarray(y, dtype=float)
    f = np.zeros_like(y)

    if beta <= 0:
        return np.full_like(y, np.nan)

    if abs(xi) < 1e-6:
        return (1 / beta) * np.exp(-y / beta)

    term = 1 + xi * y / beta
    mask = term > 0
    f[mask] = (1 / beta) * term[mask] ** (-1 / xi - 1)
    return f

def GPD_ppf(p, xi, beta):
    p = np.asarray(p, dtype=float)

    if abs(xi) < 1e-6:
        return -beta * np.log(1 - p)

    return (beta / xi) * ((1 - p) ** (-xi) - 1)

def GPD_return_level(N, u, alpha_u, obs_year, xi, beta):
    N = np.asarray(N, dtype=float)
    rate = N * obs_year * alpha_u
    z = np.full_like(N, np.nan, dtype=float)
    valid = rate > 0

    if abs(xi) < 1e-6:
        z[valid] = u + beta * np.log(rate[valid])
    else:
        z[valid] = u + (beta / xi) * (rate[valid] ** xi - 1)

    return z

def fit_GPD_excesses(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    # SciPy ajusta la GPD amb parametres (c, loc, scale). Per excedencies
    # y = x - u, fixem loc=0; c coincideix amb xi i scale amb beta.
    xi, loc, beta = genpareto.fit(y, floc=0, method="MLE")
    params = np.array([xi, beta], dtype=float)
    return params, loc, log_vers_GPD(params, y)

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

def banda_delta_GPD_return_level(N, u, alpha_u, obs_year, params, cov, alpha=0.05):
    params = np.asarray(params, dtype=float)
    N = np.asarray(N, dtype=float)
    cov = np.asarray(cov, dtype=float)

    if not np.all(np.isfinite(cov)):
        return None, None

    eps = 1e-5
    n_params = len(params)
    grad = np.zeros((len(N), n_params), dtype=float)
    h = eps * np.maximum(np.abs(params), 1.0)
    if params[1] - h[1] <= 0:
        h[1] = max(params[1] * 0.5, 1e-8)

    for j in range(n_params):
        step = np.zeros(n_params)
        step[j] = h[j]
        z_plus = GPD_return_level(N, u, alpha_u, obs_year, *(params + step))
        z_minus = GPD_return_level(N, u, alpha_u, obs_year, *(params - step))
        grad[:, j] = (z_plus - z_minus) / (2 * h[j])

    cov = (cov + cov.T) / 2
    z_hat = GPD_return_level(N, u, alpha_u, obs_year, *params)
    var = np.einsum("ij,jk,ik->i", grad, cov, grad)
    var = np.where(var >= 0, var, np.nan)
    se = np.sqrt(var)
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)

    return z_hat - z_alpha * se, z_hat + z_alpha * se

# Utilitats de dades i llindars
def mean_excess_curve(x, quantiles):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    thresholds = np.unique(np.quantile(x, quantiles))

    valid_thresholds = []
    mean_excess = []
    n_exceedances = []

    for u in thresholds:
        excesses = x[x > u] - u
        valid_thresholds.append(u)
        mean_excess.append(np.mean(excesses))
        n_exceedances.append(len(excesses))

    return np.asarray(valid_thresholds), np.asarray(mean_excess), np.asarray(n_exceedances)

# Gràfics
def ask_threshold(x, key, station_name):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    while True:
        raw = input(f"Introdueix el llindar u per a {key} ({station_name}): ").strip()
        try:
            u_input = float(raw.replace(",", "."))
        except ValueError:
            print("Cal introduir un valor numeric per al llindar.")
            continue
        u = -u_input if key in SERIES_MINIMA_NEGADA else u_input

        return u

def mean_excess_plot(x, key, station_name):
    thresholds, h_u, n_u = mean_excess_curve(x, THRESHOLD_QUANTILES)

    fig, ax = plt.subplots()
    thresholds_axis = -thresholds if key in SERIES_MINIMA_NEGADA else thresholds
    ax.plot(thresholds_axis, h_u, color="tab:orange", linewidth=2)
    ax.scatter(thresholds_axis, h_u, color="tab:orange", s=14)

    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$h(u)=\frac{1}{n_u}\sum_{x_i>u}(x_i-u)$")
    ax.grid(True, alpha=0.35)

    ax2 = ax.twinx()
    ax2.plot(thresholds_axis, n_u, color="grey", alpha=0.45, linewidth=1.5)
    ax2.set_ylabel(r"$n_u$")

    lines = list(ax.get_lines()) + list(ax2.get_lines())
    labels = ["Esperança empirica", r"$n_u$"]
    ax.legend(lines, labels, loc="best")

    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / f"mean_excess_Temp_{key}_{name}.png", dpi=300)
    return fig

def GPD_threshold_stability_plot(x, key, station_name):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    candidate_thresholds = np.unique(np.quantile(x, THRESHOLD_QUANTILES))

    hist_xi = {"u": [], "xi": [], "upper": [], "lower": []}
    hist_sigma_star = {"u": [], "sigma_star": [], "upper": [], "lower": []}
    hist_n = {"u": [], "n": []}
    z = NormalDist().inv_cdf(0.975)

    for u in candidate_thresholds:
        y = x[x > u] - u
        params_opt, _, _ = fit_GPD_excesses(y)
        ci, cov = intervals_confianca_mle(log_vers_GPD, params_opt, args=(y,), alpha=0.05)
        sigma_star = params_opt[1] - params_opt[0] * u
        grad_sigma_star = np.array([-u, 1.0])
        var_sigma_star = grad_sigma_star @ cov @ grad_sigma_star
        se_sigma_star = np.sqrt(var_sigma_star) if var_sigma_star >= 0 else np.nan

        hist_xi["u"].append(u)
        hist_xi["xi"].append(params_opt[0])
        hist_xi["upper"].append(ci[0, 1])
        hist_xi["lower"].append(ci[0, 0])

        hist_sigma_star["u"].append(u)
        hist_sigma_star["sigma_star"].append(sigma_star)
        hist_sigma_star["upper"].append(sigma_star + z * se_sigma_star)
        hist_sigma_star["lower"].append(sigma_star - z * se_sigma_star)

        hist_n["u"].append(u)
        hist_n["n"].append(len(y))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    u_xi = -np.asarray(hist_xi["u"]) if key in SERIES_MINIMA_NEGADA else hist_xi["u"]
    yerr_xi = [[e - l for e, l in zip(hist_xi["xi"], hist_xi["lower"])], [upper - e for upper, e in zip(hist_xi["upper"], hist_xi["xi"])]]
    axes[0].errorbar(u_xi, hist_xi["xi"], yerr=yerr_xi, fmt="o-", capsize=3, markersize=2, linewidth=1.2, color="tab:orange")
    axes[0].set_ylabel(r"$\xi$")
    axes[0].grid(True, alpha=0.35)

    ax_n = axes[0].twinx()
    u_n = -np.asarray(hist_n["u"]) if key in SERIES_MINIMA_NEGADA else hist_n["u"]
    ax_n.plot(u_n, hist_n["n"], color="grey", alpha=0.45, linewidth=1.2, label=r"$n_u$")
    ax_n.set_ylabel(r"$n_u$")

    u_sigma = -np.asarray(hist_sigma_star["u"]) if key in SERIES_MINIMA_NEGADA else hist_sigma_star["u"]
    yerr_sigma = [[e - l for e, l in zip(hist_sigma_star["sigma_star"], hist_sigma_star["lower"])], [upper - e for upper, e in zip(hist_sigma_star["upper"], hist_sigma_star["sigma_star"])]]
    axes[1].errorbar(u_sigma, hist_sigma_star["sigma_star"], yerr=yerr_sigma, fmt="o-", capsize=3, markersize=2, linewidth=1.2, color="green")
    axes[1].set_xlabel(r"$u$")
    axes[1].set_ylabel(r"$\sigma^*$")
    axes[1].grid(True, alpha=0.35)

    ax_n_bottom = axes[1].twinx()
    ax_n_bottom.plot(u_n, hist_n["n"], color="grey", alpha=0.45, linewidth=1.2, label=r"$n_u$")
    ax_n_bottom.set_ylabel(r"$n_u$")

    fig.tight_layout()
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_threshold_stability_Temp_{key}_{name}.png", dpi=300)
    return fig



def GPD_return_period_plot(x, dates, key, station_name, u, params, cov):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = len(x)

    xi, beta = params
    alpha_u = np.mean(x > u)
    dates_anys = pd.to_datetime(pd.Series(dates).dropna())
    obs_year = m / ((dates_anys.max() - dates_anys.min()).days / 365.25)

    N_grid = np.linspace(1e-3, 100, 400)

    z_N = GPD_return_level(N_grid, u, alpha_u, obs_year, xi, beta)
    z_low, z_high = banda_delta_GPD_return_level(N_grid, u, alpha_u, obs_year, params, cov, alpha=0.05)
    if key in SERIES_MINIMA_NEGADA:
        z_N = -z_N
        z_low, z_high = -z_high, -z_low

    fig, ax = plt.subplots()
    ax.fill_between(N_grid, z_low, z_high, color="green", alpha=0.20, label="Banda IC 95% (delta)")
    ax.plot(N_grid, z_N, color="green", linewidth=2)

    ax.set_title("")
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.set_xlabel("Periode de retorn N (anys)")
    ax.set_ylabel(r"Nivell de retorn $z_N$")
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_return_period_Temp_{key}_{name}.png", dpi=300)
    plt.close(fig)

def text_parametres_gpd_plot(key, u, n_excedencies, params, ci):
    xi, beta = params
    u_plot = -u if key in SERIES_MINIMA_NEGADA else u
    return (
        rf"$u = {u_plot:.2f}$" "\n"
        rf"$n_u = {n_excedencies}$" "\n"
        rf"$\hat{{\xi}} = {xi:.3f}$ [{ci[0, 0]:.3f}, {ci[0, 1]:.3f}]" "\n"
        rf"$\hat{{\sigma}} = {beta:.2f}$ [{ci[1, 0]:.2f}, {ci[1, 1]:.2f}]"
    )

def GPD_pdf_plot(y, u, key, station_name, params, ci, cov):
    xi_opt, beta_opt = params
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    y_vals = np.linspace(0, max(y), 300)
    pdf_vals = GPD_pdf(y_vals, xi_opt, beta_opt)
    pdf_low, pdf_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_pdf(y_vals, xi_s, beta_s), params, cov)
    if key in SERIES_MINIMA_NEGADA:
        pdf_x_vals = -(u + y_vals)
        pdf_x_hist = -(u + y)
        order_pdf = np.argsort(pdf_x_vals)
        pdf_x_vals = pdf_x_vals[order_pdf]
        pdf_vals = pdf_vals[order_pdf]
        pdf_low = pdf_low[order_pdf]
        pdf_high = pdf_high[order_pdf]
    else:
        pdf_x_vals = u + y_vals
        pdf_x_hist = u + y

    fig, ax = plt.subplots()
    ax.hist(pdf_x_hist, bins=BINS_PDF_GPD[key], density=True, alpha=0.6, color="tab:orange", edgecolor="black", label="Dades")
    ax.fill_between(pdf_x_vals, pdf_low, pdf_high, color="green", alpha=0.20, label="Banda IC 95%")
    ax.plot(pdf_x_vals, pdf_vals, color="green", linewidth=3, label="Dist. Inferida")
    ax.set_xlabel(r"Temperatura ($^\circ$C)")
    ax.set_ylabel("Densitat")
    textstr = text_parametres_gpd_plot(key, u, len(y), params, ci)
    legend_loc = "upper left" if key in SERIES_MINIMA_NEGADA else "upper right"
    legend_anchor = (0.0, 0.78) if key in SERIES_MINIMA_NEGADA else (1.0, 0.78)
    main_legend = ax.legend(loc=legend_loc)
    ax.add_artist(main_legend)
    param_handle = Line2D([], [], linestyle="none", label=textstr)
    ax.legend(handles=[param_handle], loc=legend_loc, bbox_to_anchor=legend_anchor, framealpha=0.8, handlelength=0, handletextpad=0, borderpad=0.7)
    ax.grid()
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_pdf_Temp_{key}_{name}.png", dpi=300)
    plt.close(fig)


def GPD_distribucio_plot(y, u, key, station_name, params, ci, cov):
    xi_opt, beta_opt = params
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    y_vals = np.linspace(0, max(y), 300)
    cdf_vals = GPD_cdf(y_vals, xi_opt, beta_opt)
    cdf_low, cdf_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_cdf(y_vals, xi_s, beta_s), params, cov)

    fig, ax = plt.subplots()
    ax.hist(y, bins=BINS_PDF_GPD[key], density=True, cumulative=True, alpha=0.6, color="tab:orange", edgecolor="black", label="Dades")
    ax.fill_between(y_vals, np.clip(cdf_low, 0, 1), np.clip(cdf_high, 0, 1), color="green", alpha=0.20, label="Banda IC 95%")
    ax.plot(y_vals, np.clip(cdf_vals, 0, 1), color="green", linewidth=3, label="Dist. Inferida")
    ax.set_xlabel(r"Excedència ($^\circ$C)")
    ax.set_ylabel("Densitat acumulada")
    ax.set_ylim(-0.02, 1.02)
    textstr = text_parametres_gpd_plot(key, u, len(y), params, ci)
    legend_loc = "lower right"
    legend_anchor = (1.0, 0.22)
    main_legend = ax.legend(loc=legend_loc)
    ax.add_artist(main_legend)
    param_handle = Line2D([], [], linestyle="none", label=textstr)
    ax.legend(handles=[param_handle], loc=legend_loc, bbox_to_anchor=legend_anchor, framealpha=0.8, handlelength=0, handletextpad=0, borderpad=0.7)
    ax.grid()
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_CDF_Temp_{key}_{name}.png", dpi=300)
    plt.close(fig)


def GPD_PP_plot(y, key, station_name, params, cov):
    xi_opt, beta_opt = params
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    y_sorted = np.sort(y)
    n = len(y_sorted)
    p = np.arange(1, n + 1) / (n + 1)
    F_emp = p
    if key in SERIES_MINIMA_NEGADA:
        y_for_pp = y_sorted[::-1]
        F_fit = 1 - GPD_cdf(y_for_pp, xi_opt, beta_opt)
        F_low, F_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_cdf(y_for_pp, xi_s, beta_s), params, cov)
        F_low, F_high = 1 - F_high, 1 - F_low
    else:
        y_for_pp = y_sorted
        F_fit = GPD_cdf(y_for_pp, xi_opt, beta_opt)
        F_low, F_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_cdf(y_for_pp, xi_s, beta_s), params, cov)

    fig, ax = plt.subplots()
    ax.fill_between(F_emp, F_low, F_high, color="tab:orange", alpha=0.20, label="Banda IC 95%")
    ax.plot(F_emp, F_fit, color="tab:orange", linewidth=2, label="Ajust GPD")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="y = x")
    ax.set_xlabel(r"$\hat{H}(z)$" if key in SERIES_MINIMA_NEGADA else r"$\hat{F}(y)$")
    ax.set_ylabel(r"$H(z)$" if key in SERIES_MINIMA_NEGADA else r"$F_{GPD}(y)$")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_PP_Temp_{key}_{name}.png", dpi=300)
    plt.close(fig)


def GPD_QQ_plot(y, u, key, station_name, params, cov):
    xi_opt, beta_opt = params
    name = "".join(char if char.isalnum() or char in " -_" else "_" for char in str(station_name))
    y_sorted = np.sort(y)
    n = len(y_sorted)
    p = np.arange(1, n + 1) / (n + 1)

    if key in SERIES_MINIMA_NEGADA:
        p_model = 1 - p
        q_theoretical = -(u + GPD_ppf(p_model, xi_opt, beta_opt))
        y_plot_sorted = -(u + y_sorted[::-1])
        q_low, q_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_ppf(p_model, xi_s, beta_s), params, cov)
        q_low, q_high = -(u + q_high), -(u + q_low)
    else:
        q_theoretical = u + GPD_ppf(p, xi_opt, beta_opt)
        y_plot_sorted = u + y_sorted
        q_low, q_high = banda_parametrica_hessiana(lambda xi_s, beta_s: GPD_ppf(p, xi_s, beta_s), params, cov)
        q_low, q_high = u + q_low, u + q_high

    min_val = min(np.min(q_theoretical), np.min(y_plot_sorted))
    max_val = max(np.max(q_theoretical), np.max(y_plot_sorted))
    min_val = min(min_val, np.nanmin(q_low))
    max_val = max(max_val, np.nanmax(q_high))

    fig, ax = plt.subplots()
    ax.fill_betweenx(y_plot_sorted, q_low, q_high, color="tab:orange", alpha=0.20, label="Banda IC 95%")
    ax.plot(q_theoretical, y_plot_sorted, color="tab:orange", linewidth=2, label="Ajust GPD")
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="grey", linewidth=1, label="y = x")
    ax.set_xlabel(r"Quantils inferits ($^\circ$C)")
    ax.set_ylabel(r"Quantils observats ($^\circ$C)")
    ax.legend()
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOTS_DIR / f"GPD_QQ_Temp_{key}_{name}.png", dpi=300)
    plt.close(fig)


# Taules
def parametres_gpd_taula(fila):
    u = -fila["u"] if fila["codi"] in SERIES_MINIMA_NEGADA else fila["u"]
    return {
        "serie": rf"{NOMS_SERIES_LATEX[fila['codi']]} ($u={u:.1f}\,^\circ\mathrm{{C}}$)",
        "xi": rf"${fila['xi']:.3f}\,({fila['xi_low']:.3f},\;{fila['xi_high']:.3f})$",
        "sigma": rf"${fila['sigma']:.2f}\,({fila['sigma_low']:.2f},\;{fila['sigma_high']:.2f})$",
    }

def guardar_taula_gpd_tex(resultats):
    if not resultats:
        return

    TEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = TEX_OUTPUT_DIR / "resum_gpd_entregable.tex"
    linies = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Sèrie & $n_u$ & $\hat{\xi}$ & $\hat{\tilde{\sigma}}$ \\",
        r"\hline",
        "",
    ]

    resultats_per_codi = {fila["codi"]: fila for fila in resultats}
    for codi in ORDRE_SERIES:
        if codi not in resultats_per_codi:
            continue
        fila = resultats_per_codi[codi]
        params = parametres_gpd_taula(fila)
        linies.append(" & ".join([params["serie"], str(fila["n_excedencies"]), params["xi"], params["sigma"]]) + r" \\")
        linies.append("")

    linies.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Estimacions dels paràmetres de les distribucions DPG ajustades. El llindar utilitzat en cada sèrie s'indica entre parèntesis a la primera columna. Els valors entre parèntesis a la columna de cada paràmetre corresponen als intervals de confiança del 95\%.}",
            r"\label{tab:GPD}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(linies), encoding="utf-8")
    print(f"Taula GPD guardada a: {path}")

OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.concat([pd.read_parquet(path) for path in FITXERS_PARQUET], ignore_index=True)
df["DATA"] = pd.to_datetime(df["DATA"])
grouped = df.groupby("CODI")
resultats_gpd = []

for key, station in grouped:
    station_name = station["Nom"].iloc[0]
    station_data = station[["DATA", "original"]].dropna().sort_values("DATA")
    weekly_data = station_data.groupby(pd.Grouper(key="DATA", freq=WEEKLY_MAXIMA_FREQ))["original"].max().dropna().reset_index()
    x = weekly_data["original"].values
    dates = weekly_data["DATA"].values

    selection_figures = [mean_excess_plot(x, key, station_name), GPD_threshold_stability_plot(x, key, station_name)]

    plt.show()
    u = ask_threshold(x, key, station_name)

    for fig in selection_figures:
        plt.close(fig)

    y = x[x > u] - u
    params_opt, _, nll_opt = fit_GPD_excesses(y)
    xi_opt, beta_opt = params_opt
    ci, cov = intervals_confianca_mle(log_vers_GPD, params_opt, args=(y,), alpha=0.05)

    resultats_gpd.append(
        {
            "codi": key,
            "nom": station_name,
            "u": u,
            "n_excedencies": len(y),
            "xi": xi_opt,
            "xi_low": ci[0, 0],
            "xi_high": ci[0, 1],
            "sigma": beta_opt,
            "sigma_low": ci[1, 0],
            "sigma_high": ci[1, 1],
        }
    )

    GPD_pdf_plot(y, u, key, station_name, params_opt, ci, cov)
    GPD_distribucio_plot(y, u, key, station_name, params_opt, ci, cov)
    GPD_PP_plot(y, key, station_name, params_opt, cov)
    GPD_QQ_plot(y, u, key, station_name, params_opt, cov)
    GPD_return_period_plot(x, dates, key, station_name, u, params_opt, cov)

guardar_taula_gpd_tex(resultats_gpd)
