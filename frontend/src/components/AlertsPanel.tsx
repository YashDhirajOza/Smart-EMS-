import React from 'react';
import { usePolling } from '../hooks/useLiveData';
import { AlertsResponse, fetchAlerts } from '../services/apiClient';

const AlertsPanel: React.FC = () => {
  const { data } = usePolling<AlertsResponse>(fetchAlerts, 4000);
  const alerts = data?.alerts ?? [];

  return (
    <ul className="alerts-list">
  {alerts.map((alert: AlertsResponse['alerts'][number]) => (
        <li key={alert.id} className={`alert alert-${alert.severity}`}>
          <div className="alert-header">
            <span>{alert.asset_id}</span>
            <span className="alert-severity">{alert.severity.toUpperCase()}</span>
          </div>
          <p>{alert.message}</p>
          <small>{new Date(alert.raised_at).toLocaleTimeString()}</small>
        </li>
      ))}
      {alerts.length === 0 && <li className="alert-empty">No active alerts</li>}
    </ul>
  );
};

export default AlertsPanel;
