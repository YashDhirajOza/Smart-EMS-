import React from 'react';
import { usePolling } from '../hooks/useLiveData';
import { RecommendationResponse, fetchRecommendation } from '../services/apiClient';

const AdvisoryPanel: React.FC = () => {
  const { data, error } = usePolling<RecommendationResponse>(fetchRecommendation, 8000);

  if (error) {
    return <div className="advisory-card error">Unable to load recommendation</div>;
  }

  if (!data) {
    return <div className="advisory-card loading">Waiting for recommendation...</div>;
  }

  return (
    <div className="advisory-card">
      <h3>{data.source}</h3>
      <pre className="advisory-action">{JSON.stringify(data.action, null, 2)}</pre>
      {data.confidence && <p>Confidence: {(data.confidence * 100).toFixed(1)}%</p>}
      <small>Received at {new Date(data.received_at).toLocaleTimeString()}</small>
    </div>
  );
};

export default AdvisoryPanel;
