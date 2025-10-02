import React from 'react';
import { usePolling } from '../hooks/useLiveData';
import { AssetsResponse, fetchAssets } from '../services/apiClient';

const AssetHeartbeat: React.FC = () => {
  const { data } = usePolling<AssetsResponse>(fetchAssets, 7000);
  const assets = data?.assets ?? [];

  return (
    <div className="heartbeat-container">
  {assets.map((asset: AssetsResponse['assets'][number]) => (
        <div key={asset.asset_id} className="heartbeat-item">
          <div className="heartbeat-label">
            <span>{asset.name}</span>
            <span className="heartbeat-score">{Math.round(asset.health_score ?? 0)}%</span>
          </div>
          <div className="heartbeat-graph">
            <span className="pulse" style={{ animationDuration: `${3 + Math.random() * 2}s` }} />
          </div>
        </div>
      ))}
    </div>
  );
};

export default AssetHeartbeat;
