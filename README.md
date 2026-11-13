# Rating estimation by Wasserstein barycenters — JASA reproducibility materials

This repository contains the Python code accompanying Daniel Raban, **“How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters,”** accepted for publication in the *Journal of the American Statistical Association*. A [public preprint](https://arxiv.org/abs/2410.00865) and [nontechnical explanation](https://pillowmath.github.io/Ratings/the-problem-with-averaging-ratings.html) are available online.

The paper studies an **average** rating, the **primitive** estimator (which places each user's scores on the common Wasserstein barycenter scale), and the **final rating estimator** (which additionally rescales primitive item scores to the barycenter distribution). The code includes dense and sparse implementations and experimental scripts for the paper's Section 8 and supplementary applications.

## 1. Can I preprocess my own dataset?

**Yes.** The general-purpose preprocessing script is `Build Rating Matrices/build_own_rating_matrix.py`. It takes your own **long-form CSV file** with one row per observed user–item rating and produces a ratings matrix suitable for the downstream scripts. It does not require one of the four published datasets. The dataset-specific preprocessing scripts below are also included.

For example, an input file `my_ratings.csv` might be:

```csv
user_id,item_id,rating
alice,film_a,5
alice,film_b,3
bob,film_a,4
bob,film_b,2
```

Run from the root of this repository:

```bash
python "Build Rating Matrices/build_own_rating_matrix.py" \
    --input my_ratings.csv --user-column user_id \
    --item-column item_id --rating-column rating \
    --format dense --output-dir my_analysis

cd my_analysis
python ../rating_estimator.py
```

This creates `my_analysis/rating_matrix.csv` with **users as rows, items as columns, and blank/NaN entries for missing ratings**. The estimator writes `average_ratings.csv`, `primitive_ratings.csv`, and `rating_estimator_ratings.csv` in the working directory. For larger datasets, choose `--format sparse` and run `python ../rating_estimator_sparse.py` from the output directory instead. The sparse input files are `rating_matrix.npz` and `item_index.csv` (plus `user_index.csv` for provenance).

Available preprocessing flags include `--duplicate mean|last`, `--min-user-ratings N`, and `--min-item-ratings N`. Repeated user–item pairs are reduced according to the chosen duplicate policy, and minimum-count filtering is repeated until both thresholds are met. By default all observed user–item pairs are retained. **Sparse matrices must not contain genuinely observed zero ratings:** the sparse estimation/evaluation scripts treat unstored zeros as missing. If zero is a valid rating on your scale, use the dense pipeline. In the sparse estimator's result CSVs, the score index is a **zero-based item column number**; use `item_index.csv` to recover original item IDs.

If your data are **already a user-by-item matrix**, no preprocessing script is necessary: save it as `rating_matrix.csv`, with a first index column containing user identifiers, numeric rating columns named by item, and empty entries for missing values; then run `rating_estimator.py`. All retained users and items should have at least one observed rating.

## 2. Main-paper table and figure reproduction map

This section explicitly identifies the scripts supporting **every table and figure in the main manuscript**. Run the dataset-specific preprocessing script first, then the indicated downstream script using the required input filename. The two distribution figures consist of three histograms derived from the three estimator output files; the new `plot_distribution_panels.py` combines them into one image.

| Main paper | Experiment/data | Scripts, in order | Output/action |
|---|---|---|---|
| **Figure 1 (p. 27)** | Netflix Prize rating distributions | `Build Rating Matrices/build_netflix_matrix_sparse.py` → `rating_estimator_sparse.py` → `plot_distribution_panels.py` | Three histograms of average, primitive, and final ratings, e.g. `python plot_distribution_panels.py --xmin 1 --xmax 5 --output netflix_distributions.png` |
| **Table 1 (p. 27)** | Netflix Prize 5-fold prediction errors | `Build Rating Matrices/build_netflix_matrix_sparse.py` → `prediction_comparison_sparse.py` | Prints MAE, MSE, exact-match rate, and SE for the average, primitive ranking, and five-random-ranking baseline |
| **Figure 2 (p. 28)** | Foodpanda Bangladesh rating distributions | `Build Rating Matrices/build_foodpanda_matrix.py` → `rating_estimator.py` → `plot_distribution_panels.py` | Three rating histograms, e.g. `python plot_distribution_panels.py --xmin 1 --xmax 5 --output foodpanda_distributions.png` |
| **Table 2 (p. 29)** | Foodpanda Bangladesh 5-fold prediction errors | `Build Rating Matrices/build_foodpanda_matrix.py` → `prediction_comparison.py` | Prints corresponding prediction results and SE |
| **Figure 3, left (p. 30)** | Scale-and-reverse simulation | `scale_and_reverse_simulation.py` | Interactive curve plot of loss versus number of raters |
| **Figure 3, right (p. 30)** | Quality-plus-noise simulation | `quality_plus_noise_simulation.py` | Interactive curve plot of mean squared error versus number of raters |

**Required file naming for the dataset scripts:** The Netflix builder writes `netflix_ratings_csr.npz`, `user_index.csv`, and `movie_index.csv`. In the analysis working directory, copy/rename the NPZ to `rating_matrix.npz` and `movie_index.csv` to `item_index.csv` before using `rating_estimator_sparse.py`; `prediction_comparison_sparse.py` only needs the NPZ. The Foodpanda builder writes `rating_matrix.csv` directly. The dataset-specific builders assume their raw input files are in the current working directory (or the configured `DATA_DIR`), and the analysis scripts save results to that directory. Work in separate folders for each dataset to avoid overwriting results.

**Plotting detail:** `plot_distribution_panels.py` automates the three histogram panels using the output score CSVs. The original `plot_ratings.py` is an editable **single-histogram example** (it defaults to just the final estimator, with fixed axes), not by itself an exact, one-command reproduction of the paper's composite figures. Fonts, panel sizing, and histogram bin choices may be adjusted for figure-layout fidelity. The two Figure 3 simulation scripts plot the two panels separately; combine their outputs to reproduce the two-panel layout.

**Predictive ranking detail:** The prediction-comparison scripts use *Method B (primitive ratings)* to determine an item ranking; the paper reports these results under the label *Rating Estimator*. The final estimator is a nondecreasing rescaling of the primitive scores, so their intended item ordering agrees, but ties or numerical discretization may affect exact equivalence. The scripts report **pooled standard errors across held-out rating predictions**, not fold-to-fold standard deviations.

## 3. Online-supplement tables and figures

| Supplement | Data | Scripts |
|---|---|---|
| **Figure 1, Table 1 (Section K)** | MyAnimeList | `Build Rating Matrices/build_myanimelist_matrix.py` → `rating_estimator.py` → `plot_distribution_panels.py` for Figure 1; `prediction_comparison.py` for Table 1 |
| **Figure 2, Table 2 (Section K)** | Yelp | `Build Rating Matrices/build_yelp_matrix_sparse.py` → `rating_estimator_sparse.py` → `plot_distribution_panels.py` for Figure 2; `prediction_comparison_sparse.py` for Table 2 |
| **Figure 3 (Section L)** | Four structured-missingness models | `quality_noise_sim_structured_missingness.py`; select and run each of its four named missingness settings to create the individual panels |

For Yelp, the builder writes `yelp_ratings_csr.npz` and `business_index.csv`; copy/rename them as `rating_matrix.npz` and `item_index.csv` in the separate Yelp analysis directory. By default the Yelp script selects open restaurant businesses and filters users and restaurants to at least 15 ratings (see constants in that script). The code generates each supplementary missingness panel separately, rather than automatically assembling all four panels.

A machine-readable-file explanation and dataset-field dictionary is provided in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md).

## 4. Data availability and preprocessing

Raw third-party datasets are **not redistributed** in this code archive. Their original source pages are:

- [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data). Input to Netflix preprocessing: `combined_data_*.txt` files.
- [Bangladesh Online Food Portals Customer Reviews](https://data.mendeley.com/datasets/532fxhnwbb/2). Input to Foodpanda preprocessing: `BDFoodSent-334k.csv`.
- [MyAnimeList dataset](https://www.kaggle.com/datasets/azathoth42/myanimelist). Input to MyAnimeList preprocessing: `UserAnimeList.csv`.
- [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset). Inputs: `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json`.

You are responsible for satisfying any source-dataset terms of use. Dataset version, eligibility filters, duplicate handling, and whether outputs use dense or sparse storage can affect numerical reproduction. Preprocessing details are visible in the relevant `Build Rating Matrices/*.py` files.

## 5. Installation and workflow

The accompanying Author Contributions Checklist reports **Python 3.10.5**, NumPy **2.2.6**, pandas **2.3.3**, SciPy **1.15.3**, and Matplotlib **3.10.9**. A `requirements.txt` records those reported versions (they have not all been independently reinstalled or used to rerun the full datasets here):

```bash
python -m pip install -r requirements.txt
```

1. Download the original raw dataset and run its matching preprocessing script, or use `build_own_rating_matrix.py` for your own dataset.
2. For dense inputs, run `rating_estimator.py`; for nonzero sparse inputs, run `rating_estimator_sparse.py`.
3. To produce the three distribution histograms, run `plot_distribution_panels.py` from the folder containing the estimator's CSV outputs. `plot_ratings.py` remains available for individual ad hoc histograms.
4. For rating-scale and rating-profile concordance, run `kendall_w.py` or `kendall_w_sparse.py` after the estimator; these print the W statistics described in Section 7.
5. To reproduce prediction tables, run `prediction_comparison.py` or `prediction_comparison_sparse.py` on the corresponding dataset input.
6. Run `scale_and_reverse_simulation.py` and `quality_plus_noise_simulation.py` for the simulation plots; for each supplementary missingness scenario, change the selected `rating_df` in `quality_noise_sim_structured_missingness.py` and rerun, adjusting its title accordingly.

The prediction scripts set a cross-validation seed of **42**, and the simulation scripts set seeds internally. Full dataset runtime and memory depend strongly on available hardware and dataset versions. The author checklist estimates a **10–60 minute** reproduction window, but this estimate has not been benchmarked against a complete run of all code here. Some scripts (notably the simulation plots) render interactive figures by default and may require a graphical Python environment; another option is to modify the final Matplotlib command to save to disk.

**Limitations requiring verification before claiming exact reproduction:** The supplied manuscript and simulation scripts currently differ on some stated parameter values, and the plot labeled “L2 loss” in the quality-plus-noise scripts plots the squared loss. See the editorial QA notes accompanying this revised code package; do not infer that exact published numeric outputs were independently re-generated by preparing documentation alone.

## 6. Code inventory

- `rating_estimator.py`, `rating_estimator_sparse.py`: dense/sparse Wasserstein-scale normalization, primitive and final rating scores, mean ratings.
- `plot_distribution_panels.py`, `plot_ratings.py`: three-panel figure and individual histogram plotting.
- `kendall_w.py`, `kendall_w_sparse.py`: rating-scale and rating-profile concordance statistics.
- `prediction_comparison.py`, `prediction_comparison_sparse.py`: 5-fold held-out user rating prediction evaluation and random-ranking controls.
- `scale_and_reverse_simulation.py`: scale-and-reverse simulation (main Figure 3, left).
- `quality_plus_noise_simulation.py`: quality-plus-noise simulation (main Figure 3, right).
- `quality_noise_sim_structured_missingness.py`: four supplementary structured-missingness scenarios.
- `Build Rating Matrices/build_own_rating_matrix.py`: user-supplied long-form CSV preprocessing, with configurable dense/sparse output.
- `Build Rating Matrices/build_foodpanda_matrix.py`, `build_myanimelist_matrix.py`, `build_netflix_matrix_sparse.py`, `build_yelp_matrix_sparse.py`: published dataset preprocessing.

**Function documentation:** Each Python function in this archive has a nearby comment block specifying **Inputs**, **Outputs**, and **Assumptions**. These documentation changes do not alter the scientific computations in the pre-existing scripts.

## 7. Citation

Daniel Raban, “How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters,” *Journal of the American Statistical Association* (accepted for publication). Preprint: [arXiv:2410.00865](https://arxiv.org/abs/2410.00865). Publication year, volume, pages, and DOI can be updated once available.
