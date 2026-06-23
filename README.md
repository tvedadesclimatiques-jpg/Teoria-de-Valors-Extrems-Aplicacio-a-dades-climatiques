# Teoria-de-Valors-Extrems-Aplicacio-a-dades-climatiques
Aquest repositori guarda el codi necessari per a fer l'estudi estadístic del Treball Final de Grau en Matemàtiques titulat Teoria de Valors Extrems: Aplicació a dades climàtiques.

## Analisi descriptiva de les dades

Crida:

```bash
python Analisi_dades_entregable.py
```

Genera les figures d'anàlisi de les dades.

## Ajust GEV per maxims de bloc

Crida:

```bash
python fitting_entregable.py
```

Llegeix:
- `Dades/series_extrems_fabra.parquet`
- `Dades/series_extrems_ebre.parquet`

Genera les figures dels ajustos de la distribució.

També genera la taula LaTeX per a fer el resum dels paràmetres. 

Opcionalment, tambe es pot ajustar el model no estacionari `mu(t)=mu0+mu1*t` amb la crida:

```bash
python fitting_entregable.py --mu_linear
```

Aixo genera la taula de LaTeX que agrupa els resultats.

## Ajust GPD per excedencies

Crida:

```bash
python fitting_GPD_entregable.py
```

Durant l'execucio, el programa mostra grafics per la seleccio de llindar i demana introduir manualment el valor `u` per a cada serie. Es pot escriure el decimal amb punt o coma.

Genera les figures dels ajustos de la distribució.

També genera la taula LaTeX per a fer el resum dels paràmetres. 
