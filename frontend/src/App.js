import React, { useState } from 'react';
import { Upload, FileText, BarChart3, TrendingUp, AlertCircle } from 'lucide-react';
import './App.css';

const VegaLiteChart = ({ spec }) => {
  const chartRef = React.useRef(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    if (chartRef.current && spec) {
      const renderChart = async () => {
        // Wait for vegaEmbed to be available
        let attempts = 0;
        while (!window.vegaEmbed && attempts < 50) {
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }

        if (window.vegaEmbed) {
          try {
            console.log('Rendering chart with spec:', spec);
            await window.vegaEmbed(chartRef.current, spec, {
              actions: { export: true, source: false, editor: false }
            });
            setError(null);
          } catch (err) {
            console.error('Error rendering chart:', err);
            setError(err.message);
          }
        } else {
          setError('Vega-Embed library failed to load');
        }
      };
      
      renderChart();
    }
  }, [spec]);

  if (error) {
    return (
      <div style={{ padding: '1rem', background: '#fee2e2', borderRadius: '8px' }}>
        <p style={{ color: '#991b1b', fontSize: '0.9rem' }}>
          Error rendering chart: {error}
        </p>
      </div>
    );
  }

  return <div ref={chartRef} className="chart-container" />;
};

const ChartCard = ({ chart, index }) => {
  const [showSpec, setShowSpec] = useState(false);

  return (
    <div className="chart-card">
      <div className="card-header">
        <div className="card-title-wrapper">
          <div className="card-icon">
            <BarChart3 size={24} />
          </div>
          <div>
            <h3 className="card-title">{chart.name}</h3>
            <p className="card-subtitle">Chart {index + 1}</p>
          </div>
        </div>
      </div>

      <p className="card-description">{chart.description}</p>

      <div className="card-section">
        <h4 className="section-title strengths-title">
          <TrendingUp size={16} />
          Strengths
        </h4>
        <ul className="feature-list">
          {chart.strengths?.map((strength, i) => (
            <li key={i}>{strength}</li>
          ))}
        </ul>
      </div>

      <div className="card-section">
        <h4 className="section-title weaknesses-title">
          <AlertCircle size={16} />
          Weaknesses
        </h4>
        <ul className="feature-list">
          {chart.weaknesses?.map((weakness, i) => (
            <li key={i}>{weakness}</li>
          ))}
        </ul>
      </div>

      {chart.vega_lite_spec && (
        <div className="chart-section">
          <h4 className="section-title">Visualization</h4>
          <VegaLiteChart spec={chart.vega_lite_spec} />
          
          <button
            onClick={() => setShowSpec(!showSpec)}
            className="spec-toggle-btn"
          >
            {showSpec ? 'Hide' : 'Show'} JSON Specification
          </button>
          
          {showSpec && (
            <pre className="spec-preview">
              {JSON.stringify(chart.vega_lite_spec, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};

export default function CSVAnalyzer() {
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [selectedColumns, setSelectedColumns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  React.useEffect(() => {
    // Load Vega libraries in sequence
    const loadScript = (src) => {
      return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
      });
    };

    const loadLibraries = async () => {
      try {
        await loadScript('https://cdn.jsdelivr.net/npm/vega@5');
        await loadScript('https://cdn.jsdelivr.net/npm/vega-lite@5');
        await loadScript('https://cdn.jsdelivr.net/npm/vega-embed@6');
        console.log('Vega libraries loaded successfully');
      } catch (error) {
        console.error('Error loading Vega libraries:', error);
      }
    };

    loadLibraries();
  }, []);

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === 'text/csv') {
      setFile(selectedFile);
      setError(null);
      setResults(null);
      setSelectedColumns([]);
      
      const text = await selectedFile.text();
      const lines = text.split('\n');
      if (lines.length > 0) {
        const cols = lines[0].split(',').map(col => col.trim().replace(/^["']|["']$/g, ''));
        setColumns(cols);
      }
    } else {
      setError('Please select a valid CSV file');
    }
  };

  const toggleColumn = (column) => {
    if (selectedColumns.includes(column)) {
      setSelectedColumns(selectedColumns.filter(c => c !== column));
    } else if (selectedColumns.length < 5) {
      setSelectedColumns([...selectedColumns, column]);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a CSV file');
      return;
    }
    if (selectedColumns.length < 2) {
      setError('Please select at least 2 columns');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('columns', JSON.stringify(selectedColumns));

    try {
      const response = await fetch('http://localhost:5001/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      console.log('Received data from backend:', data);
      console.log('Chart types:', data.chart_specs?.chart_types);
      
      if (data.chart_specs?.chart_types?.length === 0) {
        setError('No charts could be generated for the selected columns');
      }
      
      setResults(data);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError(err.message || 'An error occurred during analysis');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="content-wrapper">
        <div className="header">
          <h1 className="main-title">MercuryViz - AI Driven Chart Recommender</h1>
          <p className="subtitle">
            Upload your CSV file and select columns to generate chart recommendations
          </p>
        </div>

        <div className="upload-section">
          <div className="form-group">
            <label className="form-label">Upload CSV File</label>
            <div className="upload-wrapper">
              <label className="upload-box">
                <Upload size={48} className="upload-icon" />
                <span className="upload-text">
                  {file ? file.name : 'Click to upload CSV file'}
                </span>
                <input
                  type="file"
                  className="file-input"
                  accept=".csv"
                  onChange={handleFileChange}
                />
              </label>
            </div>
          </div>

          {columns.length > 0 && (
            <div className="form-group">
              <label className="form-label">Select Columns (2-5 columns)</label>
              <p className="column-count">
                Selected: {selectedColumns.length}/5
              </p>
              <div className="columns-grid">
                {columns.map((column) => (
                  <button
                    key={column}
                    onClick={() => toggleColumn(column)}
                    disabled={!selectedColumns.includes(column) && selectedColumns.length >= 5}
                    className={`column-btn ${selectedColumns.includes(column) ? 'selected' : ''} ${
                      !selectedColumns.includes(column) && selectedColumns.length >= 5 ? 'disabled' : ''
                    }`}
                  >
                    {column}
                  </button>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="error-box">
              <p>{error}</p>
            </div>
          )}

          <button
            onClick={handleAnalyze}
            disabled={loading || selectedColumns.length < 2}
            className="analyze-btn"
          >
            {loading ? (
              <>
                <div className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <FileText size={20} />
                Analyze Data
              </>
            )}
          </button>
        </div>

        {results && (
          <div>
            <div className="results-section">
              <h2 className="section-heading">Chart Recommendations</h2>
              <div className="charts-grid">
                {results.chart_specs?.chart_types?.map((chart, index) => (
                  <ChartCard key={index} chart={chart} index={index} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
