# UrjaNet Data Catalogue

This catalogue curates open-source datasets that power the UrjaNet simulator, diagnostics, and forecasting pipelines. The goal is to ground the "Living" Energy Flow dashboard and digital twin in realistic telemetry spanning solar PV, wind, energy storage, load profiles, and EV charging behaviour.

| Dataset | Scope | Suggested Usage | License / Access |
|---------|-------|-----------------|------------------|
| [Residential Load Diagrams (Mendeley)](https://data.mendeley.com/datasets/y58jknpgs8/2) | Household smart meter readings (30-min cadence) | Baseline residential demand, EV charging peaks, anomaly cases | CC BY 4.0, CSV download (requires free Mendeley account) |
| [Solar Power Generation (Kaggle)](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) | Solar PV output (AC/DC power, irradiation, temp) | Solar asset playback, solar forecast model calibration | CC0, zip download via Kaggle CLI or manual |
| [Global Solar Atlas](https://en.wikipedia.org/wiki/Global_Solar_Atlas) | Long-term solar irradiation statistics | Geo-tagged solar potential, capacity factor normalization | Public domain; raster tiles/API |
| [EV Charging Sessions (Mendeley)](https://data.mendeley.com/datasets/msxs4vj48g/1) | EVSE transaction logs with energy, duration, power | EV charger load profiles, smart alert thresholds | CC BY 4.0 |
| [Electricity Consumption & PV Production (Zenodo)](https://zenodo.org/records/6473455) | Synchronized consumption and PV generation for households | Battery charge/discharge modelling, energy flow validation | CC BY 4.0 |
| [MNRE Open Data Catalogue](https://www.data.gov.in/catalogs/?ministry=Ministry%20of%20New%20and%20Renewable%20Energy) | Policy, deployment, and resource datasets for Indian renewables | Regional calibration, asset metadata enrichment | Varies (check dataset-specific terms) |

## Integration Guidance

1. **Raw Storage**  
   Place downloaded CSV/Parquet assets under `simulator/data/raw/<dataset-name>/`. Retain README/license files alongside data for traceability.

2. **Curation Pipeline**  
   Use the notebook workspace (`simulator/notebooks/`) to explore raw data and derive curated timeseries saved to `simulator/data/processed/`. Standardize column names to match the telemetry schema (`timestamp`, `voltage`, `current`, `power_kw`, `temperature`, `state_of_charge`, `frequency`, `harmonics_thd`).

3. **Simulator Binding**  
   Update `simulator/src/asset_profiles.py` to point each asset to the curated dataset. Apply scaling factors or stochastic noise (`utils/transformers.py`) to capture variability between assets.

4. **Forecast & Diagnostics Calibration**  
   - Train simple linear projections using rolling windows from processed data to validate the digital twin assumptions.  
   - Derive threshold bands (`asset_thresholds` table) by computing percentiles or domain-specific limits per metric.  
   - For harmonics and power quality experiments, incorporate metrics from the Mendeley smart-meter dataset or any MNRE PQ datasets.

5. **License Compliance**  
   - Keep a `LICENSE.txt` or citation file per dataset inside its raw folder.  
   - Credits should surface in user-facing documentation or dashboards when applicable (e.g., footer note referencing MNRE open data).

6. **Automation Hooks**  
   Extend `scripts/validate_datasets.py` to validate new processed files (schema checks, missing values, sampling cadence). Future enhancements may automate dataset refresh via the providers' APIs.

### Download Tips

- **Kaggle (Solar Power Generation):** Install the Kaggle CLI (`pip install kaggle`), place your Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`, then run:
   - `python scripts/download_datasets.py` to download and unzip into `simulator/data/raw/kaggle/`.
- **Mendeley (Residential Loads & EV Sessions):** Requires a free Elsevier account. After accepting terms, download the ZIP manually and copy into `simulator/data/raw/mendeley/`. Document the license in a `LICENSE.txt` file.
- **Global Solar Atlas / MNRE Catalogs:** These provide bulk downloads or API queries; capture the request URL and date in a metadata note for reproducibility. When APIs are rate-limited, cache the responses in `simulator/data/raw/geo/`.

## Asset-to-Dataset Mapping (Initial Recommendation)

| Asset ID | Dataset | Notes |
|----------|---------|-------|
| `solar_panel_01`, `solar_panel_02` | Kaggle Solar Power Generation | Use different days/seasons for variation; map irradiation to expected DC output. |
| `battery_bank_01` | Zenodo PV + Consumption | Calculate SoC using net load curve; align voltage/current from dataset or derived values. |
| `ev_charger_01`, `ev_charger_02` | Mendeley EV Charging Sessions | Replay sessions with stochastic start times; add queueing logic for peak hours. |
| Additional wind assets (future) | Global Solar Atlas + MNRE wind datasets | Blend atlas irradiance or wind speed statistics with synthetic profiles. |

## Next Steps

- Script automated downloaders where APIs permit (e.g., Kaggle CLI integration in CI).
- Expand dataset coverage to include harmonics, power quality, and industrial loads.
- Store metadata (source, license, refresh cadence) in PostgreSQL `assets` table extensions for operational visibility.
