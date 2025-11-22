# CS7295_finalProj

A Python script that uses LangGraph with  **Google Gemini** to:
- Select and validate up to 5 columns from a CSV file
- Categorize columns (categorical, temporal, numerical) using few-shot prompting
- Generate Vega-Lite JSON specifications for multiple chart types
- Provide explanations of chart types, their strengths, and weaknesses

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your API key:

Google Gemini 2.5 (via Google AI Studio)**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from: https://aistudio.google.com/app/apikey

