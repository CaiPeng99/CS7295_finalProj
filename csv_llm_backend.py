"""
CSV Processor using LangGraph and Google Gemini API

This script reads a CSV file, allows users to select up to 5 columns,
categorizes them using few-shot prompting, and generates Vega-Lite JSON
visualizations with explanations of chart types and their strengths/weaknesses.

API Limits to be aware of:
- Rate Limits: Varies by tier (check Google AI Studio limits)
- Token Limits: Gemini models support large context windows (e.g., 1M+ tokens)
- Billing: Pay-per-use based on tokens (input + output)
"""

import os
import json
import pandas as pd
from typing import TypedDict, Annotated, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel




class State(TypedDict):
    messages: Annotated[list, add_messages]
    csv_data: str
    csv_df: Optional[pd.DataFrame]
    selected_columns: List[str]
    column_types: dict
    chart_specs: dict
    analysis_result: str


def estimate_tokens(text: str, model: str = "gemini-2.5-flash-lite") -> int:
    """Estimate the number of tokens in a text string."""
    # Gemini uses a rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def load_csv_dataframe(csv_path: str) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load CSV file as a pandas DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df, None
    except Exception as e:
        return None, f"Error loading CSV: {str(e)}"


def validate_columns(df: pd.DataFrame, selected_columns: List[str]) -> tuple[bool, str]:
    """Validate that selected columns exist in the DataFrame."""
    available_columns = list(df.columns)
    missing_columns = [col for col in selected_columns if col not in available_columns]
    
    if missing_columns:
        return False, f"Columns not found: {', '.join(missing_columns)}\nAvailable columns: {', '.join(available_columns)}"
    
    if len(selected_columns) <= 1:
        return False, "Please select at least two columns."
    
    if len(selected_columns) > 5:
        return False, "Please select at most five columns."
    
    return True, ""


def get_column_sample_data(df: pd.DataFrame, columns: List[str], max_rows: int = 20) -> str:
    """Get sample data for selected columns."""
    sample_df = df[columns].head(max_rows)
    return sample_df.to_string(index=False)


def create_categorization_few_shot_prompt() -> str:
    """Dataset-agnostic prompt for column categorization."""
    return """You are helping to prepare arbitrary CSV datasets for visualization.

You will be given:
- Columns: a list of column names from a CSV file.
- Data sample: a small table with example rows for those columns.

Your task:
For EACH column, assign exactly ONE of these types:

- "categorical"
  → text labels, categories, enums, IDs, institution names, course names, regions,
    values that represent groups rather than magnitudes
- "numerical"
  → real-valued or integer quantities that can be aggregated (e.g., counts, totals,
    scores, hours, percentages)
- "temporal"
  → dates, times, years, timestamps (e.g., "2024-01-01", "2023-09", "2024-01-01 10:00:00")

Special rules:
- If a column is stored as a number but represents a discrete entity
  (e.g., institution codes, IDs, categories encoded as 1, 2, 3),
  treat it as "categorical", NOT "numerical".
- Columns like "TOTAL", "Number of X", "count", "score", "revenue", "temperature"
  are usually "numerical".
- Columns like "Course Name", "Student Type", "Gender", "Institution", "Region"
  are usually "categorical".
- Date-like strings (e.g., "2024-01-01", "Jan 2024", "2023-09-05 10:30") are "temporal".

Output format:
Return ONLY valid JSON with this structure:

{
  "<column_name_1>": "<categorical|numerical|temporal>",
  "<column_name_2>": "<categorical|numerical|temporal>",
  ...
}

Rules for the output:
- Use each column name exactly as it appears in the Columns list
  (do NOT change spelling, capitalization, or spacing).
- Do NOT invent any new columns.
- Do NOT include any extra keys or text outside the JSON.
- Do NOT wrap the JSON in Markdown code fences.

Few-shot examples:

Example 1:
Columns: ["name", "age", "city"]
Data sample:
name    age  city
John    28   New York
Jane    32   Los Angeles

Response:
{"name": "categorical", "age": "numerical", "city": "categorical"}

Example 2:
Columns: ["date", "sales", "region"]
Data sample:
date        sales  region
2024-01-01  1500   North
2024-01-02  2300   South

Response:
{"date": "temporal", "sales": "numerical", "region": "categorical"}

Example 3:
Columns: ["timestamp", "temperature", "humidity"]
Data sample:
timestamp           temperature  humidity
2024-01-01 10:00:00 72.5        45.2
2024-01-01 11:00:00 73.1        46.1

Response:
{"timestamp": "temporal", "temperature": "numerical", "humidity": "numerical"}

Now analyze the given Columns and Data sample and return ONLY the JSON object.
"""


def _normalize_model_response(response) -> tuple[str, AIMessage]:
    """Normalize different model.invoke return shapes to (text, AIMessage)."""
    # model.invoke can return an AIMessage-like object, a list, or other types.
    item = None
    if isinstance(response, list):
        item = response[0] if len(response) > 0 else response
    else:
        item = response

    if hasattr(item, "content"):
        text = item.content
    else:
        text = str(item)

    ai_msg = AIMessage(content=text)
    return text, ai_msg


def categorize_columns_node(state: State, model: BaseChatModel) -> State:
    """Node that categorizes selected columns using few-shot prompting."""
    selected_columns = state.get("selected_columns", [])
    csv_data = state.get("csv_data", "")
    
    few_shot_prompt = create_categorization_few_shot_prompt()
    
    message = HumanMessage(
        content=f"{few_shot_prompt}\n\nColumns: {json.dumps(selected_columns)}\n"
                f"Data sample:\n{csv_data}\n\n"
                f"Return ONLY a JSON object with the categorization for each column."
    )
    
    # Call Google Gemini
    response = model.invoke([message])
    response_text, ai_msg = _normalize_model_response(response)
    
    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks if present)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        column_types = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object (handle nested objects)
        import re
        # Find the first { and try to match balanced braces
        brace_start = response_text.find('{')
        if brace_start != -1:
            brace_count = 0
            brace_end = brace_start
            for i in range(brace_start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i + 1
                        break
            try:
                json_str = response_text[brace_start:brace_end]
                column_types = json.loads(json_str)
            except:
                column_types = {}
                print(f"Warning: Could not parse categorization. Response: {response_text[:200]}")
        else:
            column_types = {}
            print(f"Warning: No JSON found in response. Response: {response_text[:200]}")
    
    return {
        "messages": [message, ai_msg],
        "selected_columns": selected_columns,
        "column_types": column_types,
        "csv_data": csv_data,
        "csv_df": state.get("csv_df"),
        "chart_specs": state.get("chart_specs", {}),
        "analysis_result": state.get("analysis_result", "")
    }


def create_vega_lite_few_shot_prompt() -> str:
    """Create few-shot, rule-based prompt for Vega-Lite chart generation."""
    return """You are an expert data visualization system. Your job is to recommend a small set of Vega-Lite charts for the given columns.

You MUST obey these rules:

GENERAL RULES
- Use ONLY the fields that appear in Columns.
- Column types are one of: "categorical", "temporal", "numerical".
- Never invent new fields, new derived columns, or new aggregations that are not clearly implied (e.g. no ratios, no percentages).
- If no meaningful chart can be made, return: {"chart_types": []}
- Return AT MOST 3 chart_types.
- Always include the provided Sample Data in data.values for each Vega-Lite spec.
- Do NOT include any extra text outside the JSON object.

CHART SELECTION RULES (CompassQL-style)
- Categorical + Numerical → Bar Chart or Box Plot
  - Put the categorical field on the x axis.
  - Put the numerical field on the y axis.
  - Aggregation on y (if used) should be "sum" or "mean" only.
- Temporal + Numerical → Line Chart or Area Chart
  - Put the temporal field on the x axis (type: "temporal").
  - Put the numerical field on the y axis (type: "quantitative").
  - Use a line or area mark, not bar.
- Numerical + Numerical → Scatter Plot
  - x and y should both be numerical.
  - No aggregation on x or y.
- Single Numerical Column → Histogram
  - Use bin: true on the numerical field.
  - y should be aggregate: "count".
- Two Categorical + One Numerical → Heatmap OR Grouped/Stacked Bar
  - Use one categorical on x, the other on y or color.
  - Color or height should encode the numerical field (usually aggregate: "sum").

OUTPUT FORMAT
You must return EXACTLY one JSON object with this structure:

{
  "chart_types": [
    {
      "name": "<Chart Name>",
      "description": "<1–3 sentence description of what the chart shows>",
      "strengths": ["<bullet 1>", "<bullet 2>"],
      "weaknesses": ["<bullet 1>", "<bullet 2>"],
      "vega_lite_spec": {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [/* use the Sample Data records */]},
        "mark": "<vega-lite mark, e.g. 'bar', 'line', 'point', 'rect', 'boxplot'>",
        "encoding": {
          // encodings consistent with the column types and rules above
        },
        "title": "<short, human-readable title>"
      }
    }
  ]
}

Few-shot example (for structure only):

Columns: ["city", "sales"]
Column Types: {"city": "categorical", "sales": "numerical"}

Expected (schematic) response:

{
  "chart_types": [
    {
      "name": "Bar Chart: Total Sales by City",
      "description": "Shows total sales for each city, allowing comparison of performance across locations.",
      "strengths": [
        "Easy to compare values across cities",
        "Works well with a moderate number of categories"
      ],
      "weaknesses": [
        "Can be cluttered with too many cities",
        "Does not show trends over time"
      ],
      "vega_lite_spec": {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": []},
        "mark": "bar",
        "encoding": {
          "x": {"field": "city", "type": "ordinal", "title": "City"},
          "y": {"field": "sales", "type": "quantitative", "aggregate": "sum", "title": "Total Sales"}
        },
        "title": "Total Sales by City"
      }
    }
  ]
}

Now generate chart recommendations for:"""


def generate_charts_node(state: State, model: BaseChatModel) -> State:
    """Node that generates Vega-Lite JSON specifications for different chart types."""
    selected_columns = state.get("selected_columns", [])
    column_types = state.get("column_types", {})
    csv_df = state.get("csv_df")
    
    # Get sample data values for Vega-Lite
    sample_data = []
    if csv_df is not None and len(selected_columns) > 0:
        sample_df = csv_df[selected_columns].head(100)  # Use first 100 rows for data
        sample_data = sample_df.to_dict('records')
    
    few_shot_prompt = create_vega_lite_few_shot_prompt()
    
    message = HumanMessage(
        content=f"{few_shot_prompt}\n\n"
                f"Columns: {json.dumps(selected_columns)}\n"
                f"Column Types: {json.dumps(column_types)}\n"
                f"Sample Data (first 100 rows): {json.dumps(sample_data)}\n\n"
                f"Generate 1-3 appropriate chart types with complete Vega-Lite JSON specifications. "
                f"Include the sample data in the 'values' field of each Vega-Lite spec. "
                f"Return ONLY a JSON object following the format shown above."
    )
    
    # Call Google Gemini
    response = model.invoke([message])
    response_text, ai_msg = _normalize_model_response(response)
    
    # Parse JSON response
    try:
        # Extract JSON from response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        chart_specs = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON object (handle nested objects)
        import re
        # Find the first { and try to match balanced braces
        brace_start = response_text.find('{')
        if brace_start != -1:
            brace_count = 0
            brace_end = brace_start
            for i in range(brace_start, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        brace_end = i + 1
                        break
            try:
                json_str = response_text[brace_start:brace_end]
                chart_specs = json.loads(json_str)
            except:
                chart_specs = {"chart_types": []}
                print(f"Warning: Could not parse chart specs. Response: {response_text[:500]}")
        else:
            chart_specs = {"chart_types": []}
            print(f"Warning: No JSON found in response.")
    
    # Format the analysis result
    analysis_result = format_chart_analysis(selected_columns, column_types, chart_specs)
    
    return {
        "messages": [message, ai_msg],
        "selected_columns": selected_columns,
        "column_types": column_types,
        "csv_data": state.get("csv_data", ""),
        "csv_df": csv_df,
        "chart_specs": chart_specs,
        "analysis_result": analysis_result
    }


def format_chart_analysis(columns: List[str], column_types: dict, chart_specs: dict) -> str:
    """Format the analysis result for display."""
    result = "=" * 70 + "\n"
    result += "COLUMN CATEGORIZATION\n"
    result += "=" * 70 + "\n\n"
    
    for col in columns:
        col_type = column_types.get(col, "unknown")
        result += f"  {col}: {col_type}\n"
    
    result += "\n" + "=" * 70 + "\n"
    result += "CHART RECOMMENDATIONS\n"
    result += "=" * 70 + "\n\n"
    
    chart_types = chart_specs.get("chart_types", [])
    
    for i, chart in enumerate(chart_types, 1):
        result += f"\n{i}. {chart.get('name', 'Unknown Chart Type')}\n"
        result += "-" * 70 + "\n"
        result += f"Description: {chart.get('description', 'N/A')}\n\n"
        
        strengths = chart.get('strengths', [])
        if strengths:
            result += "Strengths:\n"
            for strength in strengths:
                result += f"  • {strength}\n"
        
        weaknesses = chart.get('weaknesses', [])
        if weaknesses:
            result += "\nWeaknesses:\n"
            for weakness in weaknesses:
                result += f"  • {weakness}\n"
        
        vega_spec = chart.get('vega_lite_spec', {})
        result += f"\nVega-Lite JSON Specification:\n"
        result += json.dumps(vega_spec, indent=2)
        result += "\n\n" + "=" * 70 + "\n"
    
    return result


def create_csv_processing_graph():
    """Create a LangGraph workflow for processing CSV data with column selection."""
    
    # Initialize Google Gemini model
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        # Debug: Check if any Google-related env vars exist
        env_keys = [k for k in os.environ.keys() if 'GOOGLE' in k.upper() or 'GEMINI' in k.upper()]
        error_msg = (
            "GOOGLE_API_KEY environment variable must be set.\n"
            "Get your API key from https://aistudio.google.com/app/apikey\n"
            "Then set: export GOOGLE_API_KEY='your-api-key-here'\n"
        )
        if env_keys:
            error_msg += f"\nFound these related environment variables: {', '.join(env_keys)}"
        else:
            error_msg += "\nNo GOOGLE_API_KEY or GEMINI_API_KEY found in environment."
        raise ValueError(error_msg)
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  
        google_api_key=google_api_key
    )
    
    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("categorize_columns", lambda state: categorize_columns_node(state, model))
    workflow.add_node("generate_charts", lambda state: generate_charts_node(state, model))
    
    # Set entry point
    workflow.set_entry_point("categorize_columns")
    
    # Add edges
    workflow.add_edge("categorize_columns", "generate_charts")
    workflow.add_edge("generate_charts", END)
    
    # Compile the graph
    return workflow.compile()


def process_csv_with_column_selection(
    csv_path: str, 
    selected_columns: List[str],
    output_file: str = None
):
    """
    Main function to process a CSV file with column selection.
    
    Args:
        csv_path: Path to the CSV file
        selected_columns: List of column names to analyze (up to 5)
        output_file: Optional path to save the results
    """
    # Load CSV
    print(f"Loading CSV from: {csv_path}")
    df, error = load_csv_dataframe(csv_path)
    
    if error:
        print(error)
        return None
    
    # Validate columns
    print(f"\nValidating selected columns: {', '.join(selected_columns)}")
    is_valid, error_msg = validate_columns(df, selected_columns)
    
    if not is_valid:
        print(f"Error: {error_msg}")
        return None
    
    print("✓ All columns validated successfully")
    
    # Get sample data for selected columns
    sample_data = get_column_sample_data(df, selected_columns)
    
    # Create the graph
    print("\nCreating LangGraph workflow...")
    app = create_csv_processing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "csv_data": sample_data,
        "csv_df": df,
        "selected_columns": selected_columns,
        "column_types": {},
        "chart_specs": {},
        "analysis_result": ""
    }
    
    # Run the workflow
    print("Processing columns with Google Gemini...")
    print("  Step 1: Categorizing columns...")
    print("  Step 2: Generating chart recommendations...")
    
    try:
        result = app.invoke(initial_state)
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            print("\nError: Rate limit exceeded. Please wait a moment and try again.")
        elif "insufficient_quota" in error_msg.lower():
            print("\nError: Insufficient quota. Please check your Google AI Studio account billing.")
        elif "invalid_api_key" in error_msg.lower() or "api_key" in error_msg.lower():
            print("\nError: Invalid API key. Please check your GOOGLE_API_KEY environment variable.")
            print("Get your API key from: https://aistudio.google.com/app/apikey")
        else:
            print(f"\nError calling Google Gemini API: {error_msg}")
        raise
    
    # Extract the result
    analysis = result.get("analysis_result", "")
    chart_specs = result.get("chart_specs", {})
    
    # Display results
    print("\n" + analysis)
    
    # Save to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write(analysis)
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("FULL CHART SPECIFICATIONS (JSON)\n")
            f.write("=" * 70 + "\n\n")
            f.write(json.dumps(chart_specs, indent=2))
        print(f"\nResults saved to: {output_file}")
    
    # Also save chart specs as separate JSON file
    if output_file:
        json_file = output_file.replace('.txt', '_charts.json').replace('.json', '_charts.json')
        if not json_file.endswith('.json'):
            json_file = output_file + '_charts.json'
        with open(json_file, "w") as f:
            json.dump(chart_specs, f, indent=2)
        print(f"Chart specifications saved to: {json_file}")
    
    return {
        "analysis": analysis,
        "chart_specs": chart_specs,
        "column_types": result.get("column_types", {})
    }


if __name__ == "__main__":
    import sys
    
    # Check for CSV file path
    if len(sys.argv) < 2:
        print("Usage: python csv_llm_backend.py <csv_file_path> [column1] [column2] [column3] [column4] [column5] [output_file]")
        print("\nExample: python csv_llm_backend.py data.csv name age city region output.txt")
        print("\nNote: You can select 2-5 columns from your CSV file.")
        print("      If columns are not provided, available columns will be displayed.")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        sys.exit(1)
    
    # Load CSV to show available columns if needed
    df, error = load_csv_dataframe(csv_file)
    if error:
        print(error)
        sys.exit(1)
    
    # Get columns from command line or show available columns
    if len(sys.argv) >= 3:
        # Parse arguments: columns come first, optional output file at the end
        # Strategy: take up to 5 arguments after csv_file, check if last one is a file path
        args_after_csv = sys.argv[2:]
        selected_columns = []
        output_file = None
        
        for arg in args_after_csv:
            # Check if it's a column name
            if arg in df.columns and len(selected_columns) < 5:
                selected_columns.append(arg)
            # Check if it looks like a file path (has extension or is last arg)
            elif (arg.endswith('.txt') or arg.endswith('.json') or 
                  (len(selected_columns) > 0 and arg == args_after_csv[-1])):
                output_file = arg
                break
            else:
                # If not a column and not clearly a file, assume it's a column name anyway
                if len(selected_columns) < 5:
                    selected_columns.append(arg)
        
        if len(selected_columns) == 0:
            print("Error: No valid columns found in arguments.")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)
    else:
        # No columns provided, show available columns
        print(f"\nAvailable columns in '{csv_file}':")
        print("-" * 50)
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")
        print("-" * 50)
        print("\nPlease provide column names as arguments.")
        print("Usage: python csv_llm_backend.py <csv_file> <column1> [column2] [column3] [column4] [column5] [output_file]")
        print(f"\nExample: python csv_llm_backend.py {csv_file} {df.columns[0]} {df.columns[1] if len(df.columns) > 1 else ''}")
        sys.exit(1)
    
    # Process the CSV
    try:
        result = process_csv_with_column_selection(csv_file, selected_columns, output_file)
        if result is None:
            sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
