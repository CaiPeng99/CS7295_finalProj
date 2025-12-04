# MercuryViz - AI-Driven Chart Recommender

MercuryViz is an intelligent data visualization tool that uses Google Gemini AI to analyze CSV files and  generate chart recommendations. The system categorizes columns, analyzes column relationships, and suggests visualization types with detailed explanations.

##  Features

- **Intelligent Column Analysis**: Automatically categorizes columns as categorical, numerical, or temporal using few-shot prompting
- **Column Pairing Analysis**: Analyzes meaningful relationships between columns to suggest optimal visualizations
- **AI-Powered Chart Recommendations**: Generates multiple Vega-Lite chart specifications with:
  - Detailed descriptions
  - Strengths and weaknesses of each chart type
  - Interactive visualizations
  - JSON specifications for customization
- **Modern Web Interface**: Clean, responsive React frontend with real-time chart rendering
- **RESTful API**: Flask-based backend with comprehensive error handling and validation



### Workflow

```
User uploads CSV → Flask API → LangGraph Workflow:
  1. Categorize columns (categorical/numerical/temporal)
  2. Analyze column pairings (meaningful relationships)
  3. Generate chart recommendations (Vega-Lite specs)
  → Return JSON to frontend → Render interactive charts
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 14+ and npm
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CS7295_finalProj
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set your Google Gemini API key
export GOOGLE_API_KEY="your-api-key-here"
```

**Required Python packages:**
- `langgraph>=0.2.0`
- `langchain-google-genai>=3.0.0`
- `langchain-core>=1.0.0`
- `pandas>=2.0.0`
- `flask>=2.0.0`
- `flask-cors>=3.0.0`

### 3. Frontend Setup

```bash
cd frontend
npm install
```

## 🚀 Running the Application

### Start the Backend Server

```bash
# From the project root
python api.py
```

The backend will start on `http://localhost:5001`


### Start the Frontend Development Server

```bash
# From the frontend directory
cd frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

## 📖 Usage

1. **Upload a CSV File**: Click the upload area and select a CSV file from your computer
2. **Select Columns**: Choose 2-5 columns from your CSV to analyze
3. **Generate Recommendations**: Click "Analyze" to process your data
4. **View Charts**: Browse through AI-generated chart recommendations with:
   - Interactive Vega-Lite visualizations
   - Detailed descriptions
   - Strengths and weaknesses
   - JSON specifications for customization



## 🔧 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Required. Your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### API Configuration

The Flask server runs on:
- **Host**: `0.0.0.0` (all interfaces)
- **Port**: `5001`
- **Debug Mode**: Enabled (for development)

To change the port, modify `api.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

