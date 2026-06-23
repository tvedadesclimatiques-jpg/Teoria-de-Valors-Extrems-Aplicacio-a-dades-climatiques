# Teoria-de-Valors-Extrems-Aplicacio-a-dades-climatiques
Aquest repositori guarda el codi necessari per a fer l'estudi estadístic del Treball Final de Grau en Matemàtiques titulat Teoria de Valors Extrems: Aplicació a dades climàtiques.

## Analisi descriptiva de les dades

```bash
python Analisi_dades_entregable.py
```

Llegeix:
- `Dades/t_1880_2024_mx_mm.dat`
- `Dades/PPT_TX_TN_diari_1914-2024.txt`

Genera les figures a:
- `analisi_ebre_output_entregable/`
- `analisi_fabra_output_entregable/`

## Ajust GEV per maxims de bloc

```bash
python fitting_entregable.py
```

Llegeix:
- `Dades/series_extrems_fabra.parquet`
- `Dades/series_extrems_ebre.parquet`

Genera les figures a:
- `plots/Fittings_max_hessian_fabra_ebre/`

Genera la taula LaTeX:
- `../Teoria_de_Valors_Extrems__Aplicació_a_dades_climàtiques/resum_gev_estacionaria_entregable.tex`

Opcionalment, tambe es pot ajustar el model no estacionari `mu(t)=mu0+mu1*t`:

```bash
python fitting_entregable.py --mu_linear
```

Aixo genera tambe:
- `../Teoria_de_Valors_Extrems__Aplicació_a_dades_climàtiques/resum_mu_linear_entregable.tex`

## Ajust GPD per excedencies

```bash
python fitting_GPD_entregable.py
```

Llegeix:
- `Dades/series_extrems_fabra.parquet`
- `Dades/series_extrems_ebre.parquet`

Durant l'execucio, el programa mostra grafics de seleccio de llindar i demana introduir manualment el valor `u` per a cada serie. Es pot escriure el decimal amb punt o coma.

Genera les figures a:
- `plots/Fittings_GPD_entregable/`

Genera la taula LaTeX:
- `../Teoria_de_Valors_Extrems__Aplicació_a_dades_climàtiques/resum_gpd_entregable.tex`
