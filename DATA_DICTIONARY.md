# Data dictionary and file-format conventions

This document supplements the README for the code archive associated with
*How should we aggregate ratings? Accounting for personal rating scales via Wasserstein barycenters*.

## Canonical data structures

| File / column | Meaning | Type and missingness |
|---|---|---|
| `rating_matrix.csv` | Dense matrix of **users in rows** and **items in columns**. The first CSV column is the user identifier. Other column headers are item identifiers. | Numeric ratings. A blank cell or `NaN` means not rated; observed zero is valid in the **dense** workflow. |
| `rating_matrix.npz` | SciPy sparse user-by-item rating matrix consumed by sparse scripts. | CSR matrix; **unstored entries signify missing ratings**. Observed zero ratings cannot be represented without ambiguity by the supplied sparse workflow. |
| `user_index.csv` | Row-number-to-original-user-ID mapping for `.npz` matrices. | `index` is zero-based row number; `user_id` is original ID. |
| `item_index.csv` | Column-number-to-original-item-ID mapping for `.npz` matrices. | `index` is zero-based column number; `item_id` is original ID. |
| `average_ratings.csv` | Arithmetic mean rating, by item. | One numeric value per item. |
| `primitive_ratings.csv` | Primitive rating `R0`, after mapping individual ratings to a common Wasserstein barycenter scale. | One numeric value per item. |
| `rating_estimator_ratings.csv` | Final rating `R`, after a monotone rescaling of the primitive scores. | One numeric value per item. |

The dense estimator preserves original item-column labels in the result CSVs.
The supplied **sparse** estimator indexes result rows using zero-based column
numbers; use `item_index.csv` to recover the original item IDs. The published
Netflix and Yelp preprocessors initially name their matrix and item mapping
files `netflix_ratings_csr.npz`/`movie_index.csv` and
`yelp_ratings_csr.npz`/`business_index.csv`, respectively. Copy or rename
those files to `rating_matrix.npz`/`item_index.csv` before the generic sparse
analysis scripts.

## Generic user-data importer

Input: a long-form CSV containing one record per observed rating. The default
column names are `user_id`, `item_id`, `rating` (they are configurable by CLI
arguments); arbitrary user/item identifiers are treated as strings and ratings
must be finite numbers. Duplicate user-item records can be resolved by a mean
or the last recorded value. See README section 1 for examples and commands.

## Published raw-source fields actually used by preprocessing

| Dataset | Relevant input fields | Transformation / input assumptions |
|---|---|---|
| Netflix Prize | `combined_data_*.txt`: a movie-ID header followed by `user_id,rating,date` rows | Parse integer movie and user IDs and 1–5 ratings; date not used. Build CSR user-by-movie matrix and row/column ID lists. |
| Foodpanda Bangladesh | `BDFoodSent-334k.csv`: `reviewerName`, `name`, `ratings` | Use reviewer as user identifier and `name` as restaurant identifier; parse `ratings` (a string-encoded list of dictionaries) to extract the `restaurant_food` score. Pivot to a dense matrix; retain users/restaurants meeting the script's thresholds. |
| MyAnimeList | `UserAnimeList.csv`: `username`, `anime_id`, `my_score` | Ignore rows with `my_score=0` (unrated in this dataset); average duplicate pairs via pivot; retain users/anime meeting the script's thresholds. |
| Yelp Open Dataset | `yelp_academic_dataset_business.json`: `business_id`, `categories`, `is_open`; `yelp_academic_dataset_review.json`: `user_id`, `business_id`, `stars` | Select open restaurant businesses, parse user-business-star reviews, average duplicate reviews, apply minimum-count filters, and produce CSR matrix and ID maps. |

Source datasets may change format, version or terms. Re-run the preprocessors
with the particular raw versions cited in the paper and check their filters and
assumptions before claiming numerical equality with the published values.

## Simulation variables

| Variable | Meaning |
|---|---|
| `q_items` | Latent item-quality scores. |
| `mu` | Per-user additive shift. |
| `sigma` | Per-user multiplicative scale. |
| `eps` | User-item noise. |
| `matrix`, `rating_df` | Complete or partially observed user-item ratings, depending on script. |
| `mask` | Boolean observation indicator, true for a retained rating. |
| `user_group`, `item_group` | Two-valued group identifiers in the block-missingness model. |

**Important:** the published manuscript's noise and block-effect parameters
are not all equal to those in the uploaded simulation scripts. See the
submission quality-control checklist before claiming exact reproduction.
