CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    asset_type TEXT NOT NULL,
    name TEXT NOT NULL,
    location TEXT,
    rated_power_kw NUMERIC,
    commissioned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS asset_thresholds (
    id SERIAL PRIMARY KEY,
    asset_id TEXT REFERENCES assets(asset_id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    warn_min NUMERIC,
    warn_max NUMERIC,
    crit_min NUMERIC,
    crit_max NUMERIC
);

CREATE TABLE IF NOT EXISTS asset_health (
    asset_id TEXT PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
    health_score NUMERIC NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'critical');

CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY,
    asset_id TEXT REFERENCES assets(asset_id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    severity alert_severity NOT NULL,
    message TEXT NOT NULL,
    resolver_action JSONB,
    raised_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'rl_service',
    asset_scope JSONB,
    action JSONB,
    confidence NUMERIC,
    received_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO assets (asset_id, asset_type, name, location, rated_power_kw)
VALUES
    ('solar_panel_01', 'solar', 'Roof Solar Array A', 'Ahmedabad Campus - Block A', 12.5),
    ('solar_panel_02', 'solar', 'Roof Solar Array B', 'Ahmedabad Campus - Block B', 12.0),
    ('battery_bank_01', 'battery', 'Li-Ion Bank', 'Energy Center', 25.0),
    ('ev_charger_01', 'ev_charger', 'Charger Bay 1', 'Parking Lot', 7.2),
    ('ev_charger_02', 'ev_charger', 'Charger Bay 2', 'Parking Lot', 7.2)
ON CONFLICT (asset_id) DO NOTHING;
