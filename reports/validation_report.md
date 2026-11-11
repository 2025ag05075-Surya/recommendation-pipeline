# Data Validation Report

Severity **error** = row rejected (written to `data/errors/<table>_errors.csv`); **warning** = row kept, value repaired in the preparation stage. Full per-cell log: `data/errors/<table>_issues.csv`.

## Summary

| Table | Rows in | Valid | Rejected | Rejection rate |
|---|---:|---:|---:|---:|
| users | 2,020 | 2,000 | 20 | 0.99% |
| items | 808 | 800 | 8 | 0.99% |
| interactions | 108,951 | 103,289 | 5,662 | 5.2% |

## users -- issues by rule

| Column | Rule | Severity | Count |
|---|---|---|---:|
| user_id | duplicate_row | error | 20 |
| age | missing_value | warning | 61 |
| gender | missing_value | warning | 40 |
| age | out_of_range | warning | 13 |
| signup_date | invalid_timestamp | warning | 10 |
| age | incorrect_dtype | warning | 7 |

## items -- issues by rule

| Column | Rule | Severity | Count |
|---|---|---|---:|
| item_id | duplicate_row | error | 8 |
| category | missing_value | warning | 16 |
| genres | missing_value | warning | 16 |
| release_year | incorrect_dtype | warning | 8 |
| title | missing_value | warning | 8 |

## interactions -- issues by rule

| Column | Rule | Severity | Count |
|---|---|---|---:|
| user_id | duplicate_row | error | 1610 |
| rating | invalid_rating | error | 877 |
| item_id | referential_integrity | error | 544 |
| user_id | referential_integrity | error | 541 |
| rating | missing_value | error | 539 |
| timestamp | timestamp_out_of_range | error | 530 |
| user_id | missing_value | error | 361 |
| timestamp | invalid_timestamp_format | error | 282 |
| timestamp | missing_value | error | 280 |
| rating | incorrect_dtype | error | 218 |
