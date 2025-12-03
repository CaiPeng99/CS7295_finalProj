# # api.py
# import os
# import json
# import pandas as pd
# from flask import Flask, request, jsonify
# # Import your LangGraph functions and classes
# from csv_llm_backend import process_csv_with_column_selection

# app = Flask(__name__)
# CORS(app) # This is CRITICAL for the frontend to be allowed to call the backend

# # Create a folder for temporary file uploads
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/analyze', methods=['POST'])
# def analyze_data():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'}), 400
    
#     file = request.files['file']
#     # In a real app, you'd parse this string back to a list
#     selected_columns_str = request.form.get('columns')
    
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         # 1. Save file temporarily
#         unique_filename = f"{uuid.uuid4()}_{file.filename}"
#         filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
#         file.save(filepath)
        
#         # 2. Parse columns
#         selected_columns = json.loads(selected_columns_str)
        
#         # 3. CALL TEAMMATE'S BACKEND LOGIC
#         # Note: I'm passing None for output_file to avoid unnecessary file writes
#         result = process_csv_with_column_selection(filepath, selected_columns, output_file=None)
        
#         # 4. Clean up file
#         os.remove(filepath)
        
#         if result is None:
#             return jsonify({'error': 'Analysis failed'}), 500
            
#         # 5. Return JSON to React
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     # Run on port 5001 to match frontend
#     app.run(debug=True, port=5001)

# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import uuid
import pandas as pd                

app = Flask(__name__)
CORS(app) 

# Create a folder for temporary file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- THE MOCK FUNCTION ---
def mock_process_csv_with_column_selection(filepath, selected_columns):                
    """Simulates the backend response without calling LLM."""
    print(f"MOCK: Processing {len(selected_columns)} columns from {filepath}")                
    
    # 1. Mock Column Categorization
    column_types = {col: "numerical" if i % 2 == 0 else "categorical" for i, col in enumerate(selected_columns)}
    
    # 2. Mock Chart Specifications (The Vega-Lite JSON)
    # We create a dummy scatter plot as an example
    mock_chart_types = [                
        {
          "name": "Mock Scatter Plot (Numeric vs Numeric)",                
          "description": "This is a dummy chart for testing the connection.",                
          "strengths": ["Fast to generate", "Doesn't cost API credits"],                
          "weaknesses": ["Data is probably not meaningful", "LLM was not used"],                
          "vega_lite_spec": {                
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",                
            "data": {"values": [                
                # Mock data points based on selected columns
                {selected_columns[0]: 1, selected_columns[1]: 10},                
                {selected_columns[0]: 2, selected_columns[1]: 20},                
                {selected_columns[0]: 3, selected_columns[1]: 15}                
            ]},                
            "mark": "point",                
            "encoding": {                
              "x": {"field": selected_columns[0], "type": "quantitative"},                
              "y": {"field": selected_columns[1], "type": "quantitative"}                
            }                
          }                
        }
    ]                
    
    # Structure matches what's returned to React
    return {                
        "column_types": column_types,                
        "chart_specs": {"chart_types": mock_chart_types},                
        "analysis": "Mock analysis for testing."                
    }                

@app.route('/analyze', methods=['POST'])
def analyze_data():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    selected_columns_str = request.form.get('columns')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save file temporarily (still useful to test file handling)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        # Parse columns
        selected_columns = json.loads(selected_columns_str)                
        
        # --- CALL THE MOCK FUNCTION INSTEAD ---
        # Instead of calling the real backend logic, call the mock
        result = mock_process_csv_with_column_selection(filepath, selected_columns)                
        
        # Clean up file
        os.remove(filepath)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)