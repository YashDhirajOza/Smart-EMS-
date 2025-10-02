import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { usePolling } from '../hooks/useLiveData';
import { EnergyFlowEdge, EnergyFlowNode, EnergyFlowState, fetchEnergyFlow } from '../services/apiClient';

const width = 480;
const height = 320;

const EnergyFlowDiagram: React.FC = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const { data } = usePolling<EnergyFlowState>(fetchEnergyFlow, 5000);

  useEffect(() => {
    if (!data || !svgRef.current) {
      return;
    }
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const colorScale = d3.scaleOrdinal<string, string>(d3.schemeTableau10);

    const nodePositions = new Map<string, { x: number; y: number }>();
    data.nodes.forEach((node: EnergyFlowNode, index: number) => {
      const angle = (index / Math.max(1, data.nodes.length)) * 2 * Math.PI;
      nodePositions.set(node.node_id, {
        x: width / 2 + Math.cos(angle) * 120,
        y: height / 2 + Math.sin(angle) * 120,
      });
    });

    svg
      .append('g')
      .selectAll('line')
  .data(data.edges as EnergyFlowEdge[])
      .enter()
      .append('line')
  .attr('x1', (d: EnergyFlowEdge) => nodePositions.get(d.source)?.x ?? width / 2)
  .attr('y1', (d: EnergyFlowEdge) => nodePositions.get(d.source)?.y ?? height / 2)
  .attr('x2', (d: EnergyFlowEdge) => nodePositions.get(d.target)?.x ?? width / 2)
  .attr('y2', (d: EnergyFlowEdge) => nodePositions.get(d.target)?.y ?? height / 2)
  .attr('stroke', '#66d9ef')
  .attr('stroke-width', (d: EnergyFlowEdge) => Math.max(1, d.power_kw / 5))
      .attr('marker-end', 'url(#arrow)');

    svg
      .append('defs')
      .append('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 0 10 10')
      .attr('refX', 8)
      .attr('refY', 5)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto-start-reverse')
      .append('path')
      .attr('d', 'M 0 0 L 10 5 L 0 10 z')
      .attr('fill', '#66d9ef');

    const nodeGroup = svg.append('g');

    nodeGroup
  .selectAll('circle')
  .data(data.nodes as EnergyFlowNode[])
      .enter()
      .append('circle')
  .attr('cx', (d: EnergyFlowNode) => nodePositions.get(d.node_id)?.x ?? width / 2)
  .attr('cy', (d: EnergyFlowNode) => nodePositions.get(d.node_id)?.y ?? height / 2)
  .attr('r', 24)
  .attr('fill', (d: EnergyFlowNode) => colorScale(d.node_id));

    nodeGroup
  .selectAll('text')
  .data(data.nodes as EnergyFlowNode[])
      .enter()
      .append('text')
  .attr('x', (d: EnergyFlowNode) => nodePositions.get(d.node_id)?.x ?? width / 2)
  .attr('y', (d: EnergyFlowNode) => (nodePositions.get(d.node_id)?.y ?? height / 2) + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#ffffff')
  .text((d: EnergyFlowNode) => `${d.label} (${d.power_kw.toFixed(1)} kW)`);
  }, [data]);

  return <svg ref={svgRef} width={width} height={height} role="img" aria-label="Energy flow graph" />;
};

export default EnergyFlowDiagram;
