# UrjaNet – Intelligent Energy Management System

## Vision and Mission
UrjaNet is a cloud-native, proactive Energy Management System designed to stabilize Indian microgrids impacted by distributed renewables and rapid electrification. The platform blends high-fidelity data simulation, streaming ingestion, diagnostics, predictive modeling, and immersive visualization to deliver:

- A **Living Energy Flow** dashboard that animates real-time power movement across assets.
- A **Simplified Predictive Digital Twin** that projects near-term asset health and availability.
- A **Dynamic Smart Alert Engine** that prescribes actionable remediation steps, not just warnings.

## High-Level Architecture
- **Edge Simulator (Module 1):** Python-based data generator that replays and augments open-source datasets to emulate diverse asset behaviors, publishing JSON telemetry to MQTT every 2–5 seconds.
- **Cloud Data Backbone (Module 2):** Docker-orchestrated services (Mosquitto, InfluxDB, PostgreSQL) with an ingestion microservice that subscribes to MQTT, validates payloads, and persists time-series and metadata.
- **Diagnostics & Smart Alerts (Module 3):** Python service leveraging scikit-learn heuristics and rule-based resolvers to compute health indices (0–100%) and raise actionable alerts.
- **Predictive Digital Twin (Module 4):** Lightweight forecasting engine performing linear projections on rolling windows to estimate metrics such as battery time-to-empty or charger utilization.
- **Experience Layer (Module 5):** React + D3.js front-end delivering the animated energy flow, component heartbeats, alert center, and advisory panel consuming the backend APIs.
- **RL Integration Placeholder:** Secure endpoint to accept optimal action recommendations from a future reinforcement learning service and relay them to the advisory UI.

## Reference Data Sources
- **IoT Time-Series & Load Profiles:** [Residential Load Diagrams (Mendeley)](https://data.mendeley.com/datasets/y58jknpgs8/2), [Zenodo PV & Consumption](https://zenodo.org/records/6473455)
- **Solar & Wind Generation:** [Kaggle Solar Power Generation](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data), [Global Solar Atlas](https://en.wikipedia.org/wiki/Global_Solar_Atlas)
- **EV Charging Sessions:** [Mendeley EV Charging Dataset](https://data.mendeley.com/datasets/msxs4vj48g/1)
- **Policy & Resource Catalogues:** [MNRE Open Data](https://www.data.gov.in/catalogs/?ministry=Ministry%20of%20New%20and%20Renewable%20Energy)

See `docs/datasets.md` for detailed usage guidance, licensing notes, and asset-to-dataset mapping.

**Quick start:** configure the Kaggle CLI with an API token, then run `python scripts/download_datasets.py` to pull the sample solar dataset into `simulator/data/raw/`.

## Proposed Repository Structure
```
UrjaNet/
├── docker-compose.yml               # Orchestrates Mosquitto, InfluxDB, PostgreSQL, ingestion, API, diagnostics
├── infra/
│   ├── env/
│   │   ├── influxdb.env             # Credentials & retention config
│   │   ├── postgres.env             # Database credentials
│   │   └── api.env                  # FastAPI environment variables
│   ├── init/
│   │   ├── postgres-init.sql        # Schema + seed asset metadata
│   │   └── influxdb-init.sh         # Creates buckets, retention policies
│   └── monitoring/
│       └── grafana-dashboard.json   # Optional observability dashboard
├── simulator/
│   ├── data/
│   │   ├── solar_generation.csv     # Open dataset excerpts for solar panels
│   │   ├── battery_profile.csv      # Charge/discharge patterns
│   │   └── ev_charger_sessions.csv  # EV charging datasets
│   ├── notebooks/                   # Exploratory data understanding
│   └── src/
│       ├── main.py                  # Entry point orchestrating asset publishers
│       ├── publisher.py             # MQTT wrapper & payload schema enforcement
│       ├── dataset_loader.py        # Pandas-based dataset ingestion & interpolation
│       ├── asset_profiles.py        # Asset config (ids, scaling factors, noise)
│       └── utils/
│           ├── scheduler.py         # Async scheduling for 2–5 second publishing
│           └── transformers.py      # Feature engineering & unit conversions
├── services/
│   ├── ingestion/
│   │   ├── app.py                   # MQTT subscriber, validation, InfluxDB writer
│   │   ├── config.py                # Broker & database configuration helpers
│   │   ├── models.py                # Pydantic payload models
│   │   └── requirements.txt
│   ├── diagnostics/
│   │   ├── engine.py                # Health index computation & smart alert rules
│   │   ├── resolvers.py             # Action recommendation strategies
│   │   ├── forecast.py              # Linear projection utilities
│   │   └── requirements.txt
│   └── api/
│       ├── main.py                  # FastAPI entry point
│       ├── routers/
│       │   ├── assets.py            # Real-time/historical data endpoints
│       │   ├── alerts.py            # Smart alerts exposure
│       │   ├── digital_twin.py      # Forecast APIs
│       │   └── system.py            # RL recommendation endpoint & health ping
│       ├── dependencies.py          # Shared DB clients & auth
│       ├── schemas.py               # Pydantic response models
│       ├── services/
│       │   ├── influx_service.py    # Query helpers for measurements
│       │   ├── postgres_service.py  # Asset metadata CRUD
│       │   └── cache_service.py     # Optional Redis cache (future)
│       └── requirements.txt
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── index.tsx
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── EnergyFlowDiagram.tsx
│   │   │   ├── AssetHeartbeat.tsx
│   │   │   ├── AlertsPanel.tsx
│   │   │   └── AdvisoryPanel.tsx
│   │   ├── hooks/
│   │   │   └── useLiveData.ts
│   │   ├── services/apiClient.ts    # Typed client for FastAPI endpoints
│   │   └── styles/
│   ├── public/
│   │   └── index.html
│   └── tsconfig.json
├── docs/
│   ├── architecture.md              # Deep-dive diagrams & sequence charts
│   ├── api-contracts.md             # OpenAPI extracts and payload examples
│   └── ops-playbook.md              # Deployment & incident runbooks
├── scripts/
│   ├── load_sample_data.py          # Seeds asset metadata into PostgreSQL
│   ├── validate_datasets.py         # Ensures CSV integrity before simulation
│   └── smoke_test.sh                # End-to-end telemetry flow verification
└── Makefile                         # Common developer tasks
```

## Docker Compose Orchestration
```yaml
version: "3.9"

services:
	mosquitto:
		image: eclipse-mosquitto:2.0
		container_name: urjanet-mosquitto
		volumes:
			- ./infra/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf
			- mosquitto-data:/mosquitto/data
		ports:
			- "1883:1883"
			- "9001:9001"
		restart: unless-stopped

	influxdb:
		image: influxdb:2.7
		container_name: urjanet-influxdb
		env_file:
			- ./infra/env/influxdb.env
		volumes:
			- influxdb-data:/var/lib/influxdb2
			- ./infra/init/influxdb-init.sh:/docker-entrypoint-initdb.d/init.sh
		ports:
			- "8086:8086"
		restart: unless-stopped

	postgres:
		image: postgres:15-alpine
		container_name: urjanet-postgres
		env_file:
			- ./infra/env/postgres.env
		volumes:
			- postgres-data:/var/lib/postgresql/data
			- ./infra/init/postgres-init.sql:/docker-entrypoint-initdb.d/init.sql
		ports:
			- "5432:5432"
		restart: unless-stopped

	ingestion:
		build: ./services/ingestion
		container_name: urjanet-ingestion
		env_file:
			- ./infra/env/api.env
		depends_on:
			- mosquitto
			- influxdb
		restart: on-failure

	diagnostics:
		build: ./services/diagnostics
		container_name: urjanet-diagnostics
		env_file:
			- ./infra/env/api.env
		depends_on:
			- influxdb
			- postgres
		restart: on-failure

	api:
		build: ./services/api
		container_name: urjanet-api
		env_file:
			- ./infra/env/api.env
		ports:
			- "8000:8000"
		depends_on:
			- ingestion
			- diagnostics
			- influxdb
			- postgres
		restart: on-failure

	frontend:
		build: ./frontend
		container_name: urjanet-frontend
		environment:
			- VITE_API_BASE_URL=http://localhost:8000
		ports:
			- "5173:5173"
		depends_on:
			- api
		restart: on-failure

volumes:
	mosquitto-data:
	influxdb-data:
	postgres-data:
```

## FastAPI Endpoint Contract
| Method | Path | Description | Response Model |
|--------|------|-------------|----------------|
| `GET`  | `/api/assets` | List assets with metadata and latest health index. | `AssetListResponse` |
| `GET`  | `/api/assets/{asset_id}/realtime` | Fetch the latest telemetry snapshot across key metrics. | `TelemetrySnapshot` |
| `GET`  | `/api/assets/{asset_id}/history` | Query historical metrics with optional `start`, `end`, and `interval` parameters. | `TelemetrySeries` |
| `GET`  | `/api/energy-flow/live` | Aggregate power flows (solar → battery → loads/grid) for the Living Energy Diagram. | `EnergyFlowState` |
| `GET`  | `/api/alerts` | Retrieve active smart alerts with priorities and recommended actions. | `AlertList` |
| `GET`  | `/api/health/summary` | Summary of health indices across modules and data freshness. | `SystemHealth` |
| `GET`  | `/api/digital-twin/forecasts` | Return near-term projections (battery charge horizon, load forecast). | `ForecastBundle` |
| `POST` | `/api/system/recommendation` | Accept RL-generated optimal action payload; stores and surfaces to advisory panel. | `RecommendationAck` |
| `GET`  | `/api/system/recommendation` | Fetch the most recent recommendation for the advisory panel. | `Recommendation` |

**Security & Observability Notes**
- All endpoints guarded by JWT-based auth (with future integration to IAM).
- Rate limiting applied to write operations (`/recommendation`).
- Automatic OpenAPI docs served at `/docs` and `/redoc`.

## Database Schemas
### InfluxDB (Time-Series Telemetry)
- **Bucket:** `urjanet_telemetry`
- **Measurement:** `asset_metrics`
- **Tags:**
	- `asset_id` (e.g., `solar_panel_01`)
	- `asset_type` (`solar`, `battery`, `ev_charger`)
	- `metric_name` (`voltage`, `current`, `power_kw`, `temperature`, `state_of_charge`)
- **Fields:**
	- `value` (float)
	- `quality_score` (float, 0–1 representing confidence)
	- `anomaly_score` (float, optional output from diagnostics)
- **Retention Policy:** 30 days hot storage with continuous query into down-sampled bucket (`asset_metrics_15m`).

### PostgreSQL (Metadata & Recommendations)
- **Table:** `assets`
	- `asset_id` (PK, text)
	- `asset_type` (text)
	- `name` (text)
	- `location` (geography / text)
	- `rated_power_kw` (numeric)
	- `commissioned_at` (timestamp)
- **Table:** `asset_thresholds`
	- `id` (PK, serial)
	- `asset_id` (FK → assets.asset_id)
	- `metric_name` (text)
	- `warn_min`/`warn_max` (numeric)
	- `crit_min`/`crit_max` (numeric)
- **Table:** `asset_health`
	- `asset_id` (PK/FK)
	- `health_score` (numeric)
	- `updated_at` (timestamp)
- **Table:** `alerts`
	- `id` (PK, uuid)
	- `asset_id` (FK)
	- `metric_name` (text)
	- `severity` (enum: info, warning, critical)
	- `message` (text)
	- `resolver_action` (jsonb)
	- `raised_at` (timestamp)
	- `resolved_at` (timestamp, nullable)
- **Table:** `recommendations`
	- `id` (PK, uuid)
	- `source` (text, default `rl_service`)
	- `asset_scope` (jsonb)
	- `action` (jsonb)
	- `confidence` (numeric)
	- `received_at` (timestamp)

