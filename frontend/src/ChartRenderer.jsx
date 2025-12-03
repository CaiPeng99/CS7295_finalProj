import React, { useRef, useEffect } from 'react';
import vegaEmbed from 'vega-embed';

/**
 * Reusable component to render a Vega-Lite chart inside a card.
 */
const ChartRenderer = ({ chartData, index }) => {
    // A ref to hold the DOM element where Vega-Embed will attach the chart
    const chartContainerRef = useRef(null);

    // Run vegaEmbed whenever the chartData or index changes
    useEffect(() => {
        if (chartContainerRef.current && chartData.vega_lite_spec) {
            try {
                vegaEmbed(chartContainerRef.current, chartData.vega_lite_spec, {
                    // Optional: Disable action links (export, view source)
                    actions: false, 
                    // Optional: Set a specific rendering mode if needed
                    mode: 'vega-lite' 
                });
            } catch (error) {
                console.error("Error embedding Vega-Lite chart:", error);
            }
        }
    }, [chartData]);

    return (
        <div className="chart-card">
            <h3>{index + 1}. {chartData.name}</h3>
            
            <p><strong>Description:</strong> {chartData.description}</p>
            
            <div className="analysis-details">
                <div className="strengths">
                    <h4>👍 Strengths</h4>
                    <ul>
                        {chartData.strengths.map((s, i) => <li key={i}>{s}</li>)}
                    </ul>
                </div>
                <div className="weaknesses">
                    <h4>👎 Weaknesses</h4>
                    <ul>
                        {chartData.weaknesses.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                </div>
            </div>

            {/* The actual chart visualization container */}
            <div 
                ref={chartContainerRef} 
                className="vega-chart-container"
            >
                {/* Chart will be rendered here by vegaEmbed */}
            </div>
        </div>
    );
};

export default ChartRenderer;