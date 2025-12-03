# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import tempfile
# import os
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# UPLOAD_FOLDER = tempfile.gettempdir()
# ALLOWED_EXTENSIONS = {'csv'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def generate_mock_chart_specs(columns, column_types, sample_data):
#     """Generate mock chart specifications without calling Gemini API."""
    
#     chart_types = []
    
#     # Determine chart types based on column types
#     categorical_cols = [col for col, dtype in column_types.items() if dtype == 'categorical']
#     numerical_cols = [col for col, dtype in column_types.items() if dtype == 'numerical']
#     temporal_cols = [col for col, dtype in column_types.items() if dtype == 'temporal']
    
#     # Bar Chart - if we have categorical and numerical
#     if categorical_cols and numerical_cols:
#         chart_types.append({
#             "name": "Bar Chart",
#             "description": "A bar chart displays categorical data with rectangular bars. The height of each bar represents the value for that category.",
#             "strengths": [
#                 "Easy to compare values across categories",
#                 "Works well with many categories",
#                 "Clear visual representation of differences"
#             ],
#             "weaknesses": [
#                 "Can be cluttered with too many categories",
#                 "Not ideal for showing trends over time",
#                 "May require sorting for better insights"
#             ],
#             "vega_lite_spec": {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": sample_data},
#                 "mark": "bar",
#                 "encoding": {
#                     "x": {"field": categorical_cols[0], "type": "nominal", "title": categorical_cols[0]},
#                     "y": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
#                     "color": {"field": categorical_cols[0], "type": "nominal"}
#                 }
#             }
#         })
    
#     # Line Chart - if we have temporal and numerical
#     if temporal_cols and numerical_cols:
#         chart_types.append({
#             "name": "Line Chart",
#             "description": "A line chart connects data points with lines, ideal for showing trends and changes over time.",
#             "strengths": [
#                 "Excellent for showing trends over time",
#                 "Easy to see patterns and fluctuations",
#                 "Can display multiple series for comparison"
#             ],
#             "weaknesses": [
#                 "Not suitable for categorical data",
#                 "Can become cluttered with too many lines",
#                 "Requires ordered data (usually temporal)"
#             ],
#             "vega_lite_spec": {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": sample_data},
#                 "mark": {"type": "line", "point": True},
#                 "encoding": {
#                     "x": {"field": temporal_cols[0], "type": "temporal", "title": temporal_cols[0]},
#                     "y": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
#                     "color": {"value": "#3b82f6"}
#                 }
#             }
#         })
    
#     # Scatter Plot - if we have two numerical columns
#     if len(numerical_cols) >= 2:
#         chart_types.append({
#             "name": "Scatter Plot",
#             "description": "A scatter plot uses dots to represent values for two different numeric variables, showing relationships and correlations.",
#             "strengths": [
#                 "Shows relationships between two variables",
#                 "Can identify clusters and outliers",
#                 "Useful for correlation analysis"
#             ],
#             "weaknesses": [
#                 "Requires two numerical variables",
#                 "Can be hard to read with too many points",
#                 "Doesn't show categorical relationships well"
#             ],
#             "vega_lite_spec": {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": sample_data},
#                 "mark": "point",
#                 "encoding": {
#                     "x": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
#                     "y": {"field": numerical_cols[1], "type": "quantitative", "title": numerical_cols[1]},
#                     "color": {"value": "#8b5cf6"}
#                 }
#             }
#         })
    
#     # Histogram - for numerical distribution
#     if numerical_cols:
#         chart_types.append({
#             "name": "Histogram",
#             "description": "A histogram displays the distribution of numerical data by grouping values into bins.",
#             "strengths": [
#                 "Shows data distribution clearly",
#                 "Easy to identify central tendency",
#                 "Reveals skewness and outliers"
#             ],
#             "weaknesses": [
#                 "Only works with numerical data",
#                 "Bin size selection affects appearance",
#                 "Doesn't show individual data points"
#             ],
#             "vega_lite_spec": {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": sample_data},
#                 "mark": "bar",
#                 "encoding": {
#                     "x": {
#                         "field": numerical_cols[0],
#                         "type": "quantitative",
#                         "bin": True,
#                         "title": numerical_cols[0]
#                     },
#                     "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
#                     "color": {"value": "#10b981"}
#                 }
#             }
#         })
    
#     # Area Chart - if we have temporal and numerical
#     if temporal_cols and numerical_cols:
#         chart_types.append({
#             "name": "Area Chart",
#             "description": "An area chart is similar to a line chart but fills the area below the line, emphasizing magnitude of change over time.",
#             "strengths": [
#                 "Emphasizes magnitude of change",
#                 "Good for showing cumulative values",
#                 "Visually appealing for time series"
#             ],
#             "weaknesses": [
#                 "Can be misleading with multiple overlapping series",
#                 "Not suitable for negative values",
#                 "May obscure details in dense data"
#             ],
#             "vega_lite_spec": {
#                 "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
#                 "data": {"values": sample_data},
#                 "mark": "area",
#                 "encoding": {
#                     "x": {"field": temporal_cols[0], "type": "temporal", "title": temporal_cols[0]},
#                     "y": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
#                     "color": {"value": "#f59e0b"}
#                 }
#             }
#         })
    
#     return {"chart_types": chart_types[:5]}  # Return up to 5 chart types


# def simple_column_categorization(df, columns):
#     """Simple rule-based column categorization without AI."""
#     column_types = {}
    
#     for col in columns:
#         series = df[col]
        
#         # Check if temporal (datetime or date-like strings)
#         try:
#             pd.to_datetime(series.head(10))
#             column_types[col] = 'temporal'
#             continue
#         except:
#             pass
        
#         # Check if numerical
#         if pd.api.types.is_numeric_dtype(series):
#             column_types[col] = 'numerical'
#         else:
#             # Otherwise categorical
#             column_types[col] = 'categorical'
    
#     return column_types


# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint."""
#     return jsonify({'status': 'ok', 'mode': 'mock'})


# @app.route('/analyze', methods=['POST'])
# def analyze_csv():
#     """
#     Mock analysis endpoint - returns fake data without calling Gemini API.
#     """
#     try:
#         # Check if file is present
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
#         # Get selected columns
#         columns_json = request.form.get('columns')
#         if not columns_json:
#             return jsonify({'error': 'No columns provided'}), 400
        
#         try:
#             import json
#             selected_columns = json.loads(columns_json)
#         except json.JSONDecodeError:
#             return jsonify({'error': 'Invalid columns format'}), 400
        
#         if not isinstance(selected_columns, list):
#             return jsonify({'error': 'Columns must be an array'}), 400
        
#         if len(selected_columns) < 2:
#             return jsonify({'error': 'Please select at least 2 columns'}), 400
        
#         if len(selected_columns) > 5:
#             return jsonify({'error': 'Please select at most 5 columns'}), 400
        
#         # Save uploaded file temporarily
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Load CSV
#             df = pd.read_csv(filepath)
            
#             # Validate columns exist
#             missing_cols = [col for col in selected_columns if col not in df.columns]
#             if missing_cols:
#                 return jsonify({
#                     'error': f'Columns not found: {", ".join(missing_cols)}'
#                 }), 400
            
#             # Simple column categorization
#             column_types = simple_column_categorization(df, selected_columns)
            
#             # Get sample data (first 100 rows)
#             sample_df = df[selected_columns].head(100)
#             sample_data = sample_df.to_dict('records')
            
#             # Generate mock chart specs
#             chart_specs = generate_mock_chart_specs(selected_columns, column_types, sample_data)
            
#             # Return JSON response
#             return jsonify({
#                 'success': True,
#                 'chart_specs': chart_specs,
#                 'column_types': column_types,
#                 'selected_columns': selected_columns
#             })
            
#         finally:
#             # Clean up temporary file
#             if os.path.exists(filepath):
#                 os.remove(filepath)
    
#     except Exception as e:
#         return jsonify({'error': f'Server error: {str(e)}'}), 500


# @app.route('/columns', methods=['POST'])
# def get_columns():
#     """
#     Endpoint to get available columns from a CSV file.
#     """
#     try:
#         if 'file' not in request.files:
#             return jsonify({'error': 'No file provided'}), 400
        
#         file = request.files['file']
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         if not allowed_file(file.filename):
#             return jsonify({'error': 'Invalid file type'}), 400
        
#         # Save file temporarily
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
        
#         try:
#             # Load CSV
#             df = pd.read_csv(filepath)
            
#             # Return column names
#             return jsonify({
#                 'success': True,
#                 'columns': list(df.columns),
#                 'row_count': len(df)
#             })
            
#         finally:
#             if os.path.exists(filepath):
#                 os.remove(filepath)
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     print("=" * 70)
#     print("🎭 MOCK BACKEND SERVER (No Gemini API Required)")
#     print("=" * 70)
#     print("\n✓ This is a mock server for testing the frontend")
#     print("✓ No API keys needed")
#     print("✓ Returns fake chart recommendations based on column types")
#     print("\nStarting Flask server...")
#     print("Frontend should connect to: http://localhost:5000")
#     print("=" * 70)
#     print("\nPress Ctrl+C to stop the server\n")
    
#     app.run(debug=True, host='0.0.0.0', port=5001)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import tempfile
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_mock_chart_specs(columns, column_types, sample_data):
    """Generate mock chart specifications without calling Gemini API."""
    
    chart_types = []
    
    print(f"Generating charts for columns: {columns}")
    print(f"Column types: {column_types}")
    print(f"Sample data length: {len(sample_data)}")
    
    # Determine chart types based on column types
    categorical_cols = [col for col, dtype in column_types.items() if dtype == 'categorical']
    numerical_cols = [col for col, dtype in column_types.items() if dtype == 'numerical']
    temporal_cols = [col for col, dtype in column_types.items() if dtype == 'temporal']
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    print(f"Temporal columns: {temporal_cols}")
    
    # Bar Chart - if we have categorical and numerical
    if categorical_cols and numerical_cols:
        print("Adding Bar Chart")
        chart_types.append({
            "name": "Bar Chart",
            "description": "A bar chart displays categorical data with rectangular bars. The height of each bar represents the value for that category.",
            "strengths": [
                "Easy to compare values across categories",
                "Works well with many categories",
                "Clear visual representation of differences"
            ],
            "weaknesses": [
                "Can be cluttered with too many categories",
                "Not ideal for showing trends over time",
                "May require sorting for better insights"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": "bar",
                "encoding": {
                    "x": {"field": categorical_cols[0], "type": "nominal", "title": categorical_cols[0]},
                    "y": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
                    "color": {"field": categorical_cols[0], "type": "nominal"}
                },
                "width": 400,
                "height": 300
            }
        })
    
    # Scatter Plot - if we have two numerical columns
    if len(numerical_cols) >= 2:
        print("Adding Scatter Plot")
        chart_types.append({
            "name": "Scatter Plot",
            "description": "A scatter plot uses dots to represent values for two different numeric variables, showing relationships and correlations.",
            "strengths": [
                "Shows relationships between two variables",
                "Can identify clusters and outliers",
                "Useful for correlation analysis"
            ],
            "weaknesses": [
                "Requires two numerical variables",
                "Can be hard to read with too many points",
                "Doesn't show categorical relationships well"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": "point",
                "encoding": {
                    "x": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
                    "y": {"field": numerical_cols[1], "type": "quantitative", "title": numerical_cols[1]},
                    "color": {"value": "#8b5cf6"},
                    "size": {"value": 60}
                },
                "width": 400,
                "height": 300
            }
        })
    
    # Histogram - for numerical distribution
    if numerical_cols:
        print("Adding Histogram")
        chart_types.append({
            "name": "Histogram",
            "description": "A histogram displays the distribution of numerical data by grouping values into bins.",
            "strengths": [
                "Shows data distribution clearly",
                "Easy to identify central tendency",
                "Reveals skewness and outliers"
            ],
            "weaknesses": [
                "Only works with numerical data",
                "Bin size selection affects appearance",
                "Doesn't show individual data points"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": "bar",
                "encoding": {
                    "x": {
                        "field": numerical_cols[0],
                        "type": "quantitative",
                        "bin": True,
                        "title": numerical_cols[0]
                    },
                    "y": {"aggregate": "count", "type": "quantitative", "title": "Count"},
                    "color": {"value": "#10b981"}
                },
                "width": 400,
                "height": 300
            }
        })
    
    # Line Chart - if we have temporal and numerical (or just 2+ numerical)
    if temporal_cols and numerical_cols:
        print("Adding Line Chart (temporal)")
        chart_types.append({
            "name": "Line Chart",
            "description": "A line chart connects data points with lines, ideal for showing trends and changes over time.",
            "strengths": [
                "Excellent for showing trends over time",
                "Easy to see patterns and fluctuations",
                "Can display multiple series for comparison"
            ],
            "weaknesses": [
                "Not suitable for categorical data",
                "Can become cluttered with too many lines",
                "Requires ordered data (usually temporal)"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": temporal_cols[0], "type": "temporal", "title": temporal_cols[0]},
                    "y": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
                    "color": {"value": "#3b82f6"}
                },
                "width": 400,
                "height": 300
            }
        })
    elif len(numerical_cols) >= 2:
        # If no temporal but have 2 numerical, make a line chart with first as x-axis
        print("Adding Line Chart (numerical)")
        chart_types.append({
            "name": "Line Chart",
            "description": "A line chart connects data points with lines, showing the relationship between two numerical variables.",
            "strengths": [
                "Shows trends and patterns",
                "Easy to see relationships",
                "Good for ordered numerical data"
            ],
            "weaknesses": [
                "Assumes order in x-axis",
                "May not be meaningful without temporal data",
                "Can be misleading with unordered data"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": {"type": "line", "point": True},
                "encoding": {
                    "x": {"field": numerical_cols[0], "type": "quantitative", "title": numerical_cols[0]},
                    "y": {"field": numerical_cols[1], "type": "quantitative", "title": numerical_cols[1]},
                    "color": {"value": "#3b82f6"}
                },
                "width": 400,
                "height": 300
            }
        })
    
    # If we have categorical columns, add a pie chart equivalent (arc chart)
    if categorical_cols and numerical_cols:
        print("Adding Pie Chart")
        chart_types.append({
            "name": "Pie Chart",
            "description": "A pie chart shows the proportion of categories as slices of a circle.",
            "strengths": [
                "Shows proportions clearly",
                "Easy to understand at a glance",
                "Good for showing parts of a whole"
            ],
            "weaknesses": [
                "Hard to compare similar values",
                "Not good with many categories",
                "Can be misleading with certain data"
            ],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": sample_data},
                "mark": {"type": "arc", "tooltip": True},
                "encoding": {
                    "theta": {"field": numerical_cols[0], "type": "quantitative", "aggregate": "sum"},
                    "color": {"field": categorical_cols[0], "type": "nominal"}
                },
                "width": 400,
                "height": 300
            }
        })
    
    print(f"Generated {len(chart_types)} chart types")
    return {"chart_types": chart_types[:5]}  # Return up to 5 chart types


def simple_column_categorization(df, columns):
    """Simple rule-based column categorization without AI."""
    column_types = {}
    
    for col in columns:
        series = df[col]
        
        # Check if numerical first (most important)
        if pd.api.types.is_numeric_dtype(series):
            column_types[col] = 'numerical'
            continue
        
        # Check if temporal (datetime or date-like strings)
        try:
            pd.to_datetime(series.head(10), errors='raise')
            column_types[col] = 'temporal'
            continue
        except:
            pass
        
        # Otherwise categorical
        column_types[col] = 'categorical'
    
    print(f"Column categorization: {column_types}")
    return column_types


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'mode': 'mock'})


@app.route('/analyze', methods=['POST'])
def analyze_csv():
    """
    Mock analysis endpoint - returns fake data without calling Gemini API.
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Get selected columns
        columns_json = request.form.get('columns')
        if not columns_json:
            return jsonify({'error': 'No columns provided'}), 400
        
        try:
            import json
            selected_columns = json.loads(columns_json)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid columns format'}), 400
        
        if not isinstance(selected_columns, list):
            return jsonify({'error': 'Columns must be an array'}), 400
        
        if len(selected_columns) < 2:
            return jsonify({'error': 'Please select at least 2 columns'}), 400
        
        if len(selected_columns) > 5:
            return jsonify({'error': 'Please select at most 5 columns'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Validate columns exist
            missing_cols = [col for col in selected_columns if col not in df.columns]
            if missing_cols:
                return jsonify({
                    'error': f'Columns not found: {", ".join(missing_cols)}'
                }), 400
            
            # Simple column categorization
            column_types = simple_column_categorization(df, selected_columns)
            
            # Get sample data (first 100 rows)
            sample_df = df[selected_columns].head(100)
            sample_data = sample_df.to_dict('records')
            
            # Generate mock chart specs
            chart_specs = generate_mock_chart_specs(selected_columns, column_types, sample_data)
            
            # Return JSON response
            return jsonify({
                'success': True,
                'chart_specs': chart_specs,
                'column_types': column_types,
                'selected_columns': selected_columns
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/columns', methods=['POST'])
def get_columns():
    """
    Endpoint to get available columns from a CSV file.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Return column names
            return jsonify({
                'success': True,
                'columns': list(df.columns),
                'row_count': len(df)
            })
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🎭 MOCK BACKEND SERVER (No Gemini API Required)")
    print("=" * 70)
    print("\n✓ This is a mock server for testing the frontend")
    print("✓ No API keys needed")
    print("✓ Returns fake chart recommendations based on column types")
    print("\nStarting Flask server...")
    print("Frontend should connect to: http://localhost:5000")
    print("=" * 70)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)