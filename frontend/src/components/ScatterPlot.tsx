import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { DatasetResponse } from '../types/api';

interface ScatterPlotProps {
  data: DatasetResponse | null;
  predictions: number[] | null;
  meshGrid: [number, number][] | null;
  gridSize: number;
}

export default function ScatterPlot({ data, predictions, meshGrid, gridSize }: ScatterPlotProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const width = 550;
  const height = 550;

  useEffect(() => {
    if (!data) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Cálculo dos limites do gráfico
    const allX = [...data.X_train, ...data.X_test];
    const xExtent = d3.extent(allX, d => d[0]) as [number, number];
    const yExtent = d3.extent(allX, d => d[1]) as [number, number];

    const xMargin = (xExtent[1] - xExtent[0]) * 0.1 || 0.1;
    const yMargin = (yExtent[1] - yExtent[0]) * 0.1 || 0.1;

    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - xMargin, xExtent[1] + xMargin])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - yMargin, yExtent[1] + yMargin])
      .range([height, 0]);

    const colorScale = d3.scaleOrdinal<number, string>()
      .domain([0, 1])
      .range(["#f59322", "#0877bd"]);

    // Pintura das fronteiras de decisão no Canvas
    const canvas = canvasRef.current;
    if (canvas && predictions && meshGrid && meshGrid.length > 0) {
      const context = canvas.getContext('2d');
      if (context) {
        context.clearRect(0, 0, width, height);
        
        const rectWidth = width / gridSize;
        const rectHeight = height / gridSize;

        predictions.forEach((pred, i) => {
          const pt = meshGrid[i];
          context.fillStyle = pred === 1 ? 'rgba(8, 119, 189, 0.3)' : 'rgba(245, 147, 34, 0.3)';
          // Ajuste matemático de +1 na dimensão evita falhas de antialiasing (linhas brancas)
          context.fillRect(xScale(pt[0]), yScale(pt[1]) - rectHeight, rectWidth + 1, rectHeight + 1); 
        });
      }
    }

    // Desenho dos eixos cartesianos no SVG
    const xAxis = d3.axisBottom(xScale).ticks(8);
    const yAxis = d3.axisLeft(yScale).ticks(8);

    svg.append("g")
       .attr("transform", `translate(0,${height})`)
       .call(xAxis);
       
    svg.append("g")
       .call(yAxis);

    // Plotagem das amostras de Treino (borda branca fina)
    svg.append("g")
      .selectAll("circle")
      .data(data.X_train)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d[0]))
      .attr("cy", d => yScale(d[1]))
      .attr("r", 4)
      .style("fill", (d, i) => colorScale(data.y_train[i]))
      .style("stroke", "white")
      .style("stroke-width", 0.5);

    // Plotagem das amostras de Teste (borda preta espessa)
    svg.append("g")
      .selectAll("circle")
      .data(data.X_test)
      .enter()
      .append("circle")
      .attr("cx", d => xScale(d[0]))
      .attr("cy", d => yScale(d[1]))
      .attr("r", 4)
      .style("fill", (d, i) => colorScale(data.y_test[i]))
      .style("stroke", "black")
      .style("stroke-width", 1.5);

  }, [data, predictions, meshGrid, gridSize]);

  return (
    <div style={{ position: 'relative', width: width, height: height, margin: '0 auto' }}>
      <canvas 
        ref={canvasRef} 
        width={width} 
        height={height} 
        style={{ position: 'absolute', top: 0, left: 0 }}
      />
      <svg 
        ref={svgRef} 
        width={width} 
        height={height} 
        style={{ position: 'absolute', top: 0, left: 0, overflow: 'visible' }}
      />
    </div>
  );
}