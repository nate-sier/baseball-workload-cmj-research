# Baseball Game Workload & CMJ — Fresh Repository

**This is a new, standalone GitHub/Streamlit project.** No legacy `v6/` folders, SQL access, or `snapshot_builder` imports. The only Streamlit entry point is **`app.py` at repository root**.

## What it does

Connects directly to the **Python Reports** Google Spreadsheet, reading the `PP_Sprint` and `Jump Data` worksheets. Produces cumulative game workload (1, 3, 7, 14, 21, 28 calendar days before each jump test), fits athlete-random-intercept and random-workload-slope mixed models of **raw jump values**, and describes workload before acute drops of at least 5% in CI, CI100, and mRSI. Offers an Excel-upload alternative.

The model is fitted independently for each window, with within-athlete and between-athlete workload effects, month controls, and an optional test-gap control. It reports original-unit and within-athlete standardized effects, confidence intervals, p-values, random-slope variance, AIC comparisons, marginal/conditional R², and a residual lag-1 check. Model fit failure is visibly reported, not replaced with a different model.

## NEW GitHub repository: exact steps

1. Extract the downloaded ZIP. The files in this directory are the **contents of the new repository**.
2. Go to **https://github.com/new**. Name the repository `baseball-workload-cmj-research`, choose **Private** (athlete data), and leave **Add a README** and **Add .gitignore** unchecked. Click **Create repository**.
3. Upload **all contents of this folder into the top level of your new repo**, including the `.streamlit` folder and `.gitignore`. Do **not** upload an enclosing folder such as `baseball-workload-research-fresh/` as one level of nesting. GitHub's `Add file > Upload files` works, or use Git from Terminal.
4. Verify the repository home page lists **`app.py`, `data_pipeline.py`, `analysis.py`, `requirements.txt`** at the top level.
5. Go to **https://share.streamlit.io** and **Create app > Deploy a public or private app from GitHub**. Select your **new** repository and the `main` branch. For **Main file path**, enter **`app.py`**, NOT a `v6/` subfolder file. Use Python 3.12 when this is configurable.
6. Go to your new Streamlit app's **Settings > Secrets**, and paste the private secrets described below. Save. Reboot the app.
7. Click **Load / refresh source data** in the sidebar, then **Fit all six mixed-effects models**.

If Streamlit is still showing a traceback from `season_jump_workload_dashboard_v6`, you opened the **old app**, not this newly deployed one. Create the app from the new repository and ensure `app.py` is the main file.

**Cloud privacy:** Streamlit Community Cloud normally makes an app from a private repo private by default. Check the actual deployment's visibility/access permissions before using it with proprietary athlete data. Streamlit's free tier limits the number of private apps; if the platform prevents a second private deployment, remove the old private deployment (not the GitHub repo) first or use approved private hosting. Protect the spreadsheet and get organizational approval for any external hosting.

## Google Sheets connection (required for auto pull)

The app reads both worksheets from the same Google Sheet via a Google **service account**. You must provide private credentials as Streamlit **Secrets** — GitHub cannot authenticate by itself.

Your service account must have at least **Viewer** access to the Python Reports spreadsheet. If the service account used in your old dashboard already has access, reuse it.

In Streamlit Cloud **App settings > Secrets**, paste **TOML** containing:

```toml
sheet_id = "PASTE_ID_FROM_PYTHON_REPORTS_SHEET_URL"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "...@....iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

**Easiest for your existing Mac connection:** If your working service-account file is `~/Desktop/service_account.json`, run this once *locally*, from the extracted project folder:

```bash
python3 setup_secrets.py --sheet-id "YOUR_PYTHON_REPORTS_SHEET_ID"
```

This produces **`.streamlit/secrets.toml`** with the right TOML syntax. It's **gitignored**. Copy its contents into the **Secrets field in Streamlit Cloud**. Never copy secrets into GitHub, screenshots, or this chat.

The sheet ID is the segment after `/spreadsheets/d/` and before `/edit` in the Google Sheet URL. If you previously set `PYTHON_REPORTS_SHEET_ID`, the app accepts that environment variable when running locally.

## Run locally

Unzip the package first, then navigate into the extracted folder containing `app.py` (Terminal must NOT be in `~`):

```bash
cd ~/Downloads/baseball-workload-research-fresh
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

You can also use a `Python Reports.xlsx` upload in the app instead of Sheets if you do not have service-account permissions.

## File layout

```text
app.py                        # ONLY Streamlit Cloud entry point
analysis.py                   # mixed model + event comparisons
data_pipeline.py              # PP_Sprint and Jump Data processing
requirements.txt             # Python dependencies
.python-version              # Python 3.12 target
setup_secrets.py              # optional local Google credentials conversion
secrets.example.toml         # fake values only
.gitignore                   # data / secrets exclusion
.streamlit/config.toml       # light theme
tests/test_pipeline.py        # synthetic tests
```

There is **no real game data, player names, sheet ID, or credential material** in this repository.

## PP_Sprint input columns

`batter`, `mlbam_game_pk`, `game_date`, `max_effort_runs`, `max_effort_distance_covered_yards`, `distance_covered_yards`. Other monthly fields are ignored. Games are deduplicated by athlete+date+game PK and doubleheaders add their workloads. `Jump Data` requires an athlete name, jump date, and concentric impulse values. CI100 and mRSI are optional and shown only when found.

## Important research assumptions

- The **jump-test day is excluded** from workload windows. Missing calendar days *between recorded games* are zero **recorded game load**, not confirmed off-days.
- The analysis includes athlete-days with a full 28-day pre-test history from their **first recorded game** and no later than the day after their **last recorded game**. This is a coverage restriction, not proof the sheet is complete.
- Athlete matching uses normalized **names**, not a verified shared MLBAM athlete identifier; use the optional `pp_batter,jump_athlete` CSV for mismatches. Review counts and consider manual identity validation.
- Multiple trials on one date are averaged into one outcome. Same-day trial variability is **not** day-to-day test–retest typical error.
- A **random slope** accommodates between-athlete variation in workload associations but does not correct temporal autocorrelation. Inspect lag-1 residual correlations; this app does **not** fit an AR(1) error structure.
- Adjacent cumulative windows overlap; don't interpret their p-values as independent findings. Pre-specify primary workload/outcome and address multiple comparisons.
- Acute ≥5% declines are exploratory descriptive flags; the threshold is not validated as a meaningful change solely because it is 5%.

## Troubleshooting

- **`ModuleNotFoundError: data_pipeline`**: confirm `app.py` and `data_pipeline.py` are siblings in the repo root. Streamlit's Main file path must be `app.py`.
- **Still seeing `season_jump_workload_dashboard_v6` in the traceback**: you deployed the old repo or old app, not the new repo.
- **`SpreadsheetNotFound` / permission error**: Share the Python Reports Google Sheet with the service-account `client_email` as a Viewer; confirm the Sheet ID in Secrets.
- **No matched players**: compare `batter` names vs `Athlete` names, and upload a `pp_batter,jump_athlete` mapping CSV.
- **Many missing game load cells**: do not automatically replace with zeros unless you know blanks mean zero.
- **No eligible tests after 28-day run-in**: confirm dates and enough recorded game coverage for each athlete.
- **Slow fitting**: Random slope models for all six windows can be expensive; run on demand with the button and inspect failures or warnings.
