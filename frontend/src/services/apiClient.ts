const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      Authorization: `Bearer ${import.meta.env.VITE_API_TOKEN ?? 'development-secret'}`,
    },
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export interface EnergyFlowNode {
  node_id: string;
  label: string;
  power_kw: number;
}

export interface EnergyFlowEdge {
  source: string;
  target: string;
  power_kw: number;
}

export interface EnergyFlowState {
  generated_at: string;
  nodes: EnergyFlowNode[];
  edges: EnergyFlowEdge[];
}

export interface AlertsResponse {
  alerts: Array<{
    id: string;
    asset_id: string;
    metric_name: string;
    severity: string;
    message: string;
    resolver_action: Record<string, unknown>;
    raised_at: string;
  }>;
}

export interface RecommendationResponse {
  id: string;
  source: string;
  asset_scope: Record<string, unknown>;
  action: Record<string, unknown>;
  confidence?: number;
  received_at: string;
}

export interface AssetsResponse {
  assets: Array<{
    asset_id: string;
    asset_type: string;
    name: string;
    location?: string;
    rated_power_kw?: number;
    health_score?: number;
  }>;
}

export async function fetchEnergyFlow(): Promise<EnergyFlowState> {
  return fetchJson('/api/energy-flow/live');
}

export async function fetchAlerts(): Promise<AlertsResponse> {
  return fetchJson('/api/alerts');
}

export async function fetchRecommendation(): Promise<RecommendationResponse> {
  return fetchJson('/api/system/recommendation');
}

export async function fetchAssets(): Promise<AssetsResponse> {
  return fetchJson('/api/assets');
}
