"""
Flask API for CSV Analysis with LLM Backend

This API serves the frontend React application and uses the real LLM backend
(csv_llm_backend.py) powered by Google Gemini to analyze CSV files and generate
chart recommendations.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import uuid
import traceback

# Import the real LLM backend
from csv_llm_backend import process_csv_with_column_selection

app = Flask(__name__)
CORS(app)  # This is CRITICAL for the frontend to be allowed to call the backend

# Create a folder for temporary file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'mode': 'real_llm',
        'message': 'Backend is ready to process CSV files with LLM'
    })


@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Analyze CSV file and generate chart recommendations using LLM backend.
    
    Expected form data:
    - file: CSV file upload
    - columns: JSON string array of column names to analyze (2-5 columns)
    
    Returns:
    - JSON response with column_types, chart_specs, and analysis
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    selected_columns_str = request.form.get('columns')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
    
    # Validate columns parameter
    if not selected_columns_str:
        return jsonify({'error': 'No columns provided'}), 400
    
    filepath = None
    try:
        # Parse columns JSON
        try:
            selected_columns = json.loads(selected_columns_str)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid columns format. Expected JSON array'}), 400
        
        if not isinstance(selected_columns, list):
            return jsonify({'error': 'Columns must be an array'}), 400
        
        if len(selected_columns) < 2:
            return jsonify({'error': 'Please select at least 2 columns'}), 400
        
        if len(selected_columns) > 5:
            return jsonify({'error': 'Please select at most 5 columns'}), 400
        
        # Save file temporarily
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"Processing CSV: {file.filename}")
        print(f"Selected columns: {', '.join(selected_columns)}")
        print(f"{'='*70}\n")
        
        # Call the REAL LLM backend
        result = process_csv_with_column_selection(
            csv_path=filepath,
            selected_columns=selected_columns,
            output_file=None  # Don't save to file, just return JSON
        )
        
        if result is None:
            return jsonify({
                'error': 'Analysis failed. Please check the server logs for details.'
            }), 500
        
        # Ensure chart_specs has the right structure
        if 'chart_specs' not in result:
            result['chart_specs'] = {'chart_types': []}
        elif 'chart_types' not in result['chart_specs']:
            result['chart_specs'] = {'chart_types': []}
        
        print(f"\n✓ Analysis complete. Generated {len(result.get('chart_specs', {}).get('chart_types', []))} chart recommendations.")
        
        return jsonify(result)
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error during analysis: {error_msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        
        # Provide helpful error messages
        if "GOOGLE_API_KEY" in error_msg or "API key" in error_msg:
            return jsonify({
                'error': 'Google API key not configured. Please set GOOGLE_API_KEY environment variable.'
            }), 500
        elif "rate_limit" in error_msg.lower() or "429" in error_msg:
            return jsonify({
                'error': 'API rate limit exceeded. Please wait a moment and try again.'
            }), 429
        elif "quota" in error_msg.lower():
            return jsonify({
                'error': 'API quota exceeded. Please check your Google AI Studio account.'
            }), 500
        else:
            return jsonify({
                'error': f'Analysis failed: {error_msg}'
            }), 500
    
    finally:
        # Clean up temporary file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {filepath}: {e}")


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CSV Analysis API with LLM Backend")
    print("=" * 70)
    print("\n✓ Using real LLM backend (Google Gemini)")
    print("✓ API endpoint: http://localhost:5001/analyze")
    print("✓ Health check: http://localhost:5001/health")
    print("\n⚠️  Make sure GOOGLE_API_KEY environment variable is set!")
    print("   Get your API key from: https://aistudio.google.com/app/apikey")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)