import os
import json
import pandas as pd
from typing import TypedDict, Annotated, List, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class State(TypedDict):
    messages: Annotated[list, add_messages]
    csv_data: str
    csv_df: Optional[pd.DataFrame]
    selected_columns: List[str]
    column_types: dict
    chart_specs: dict
    analysis_result: str


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
    
    if len(selected_columns) < 2:
        return False, "Please select at least two columns."
    
    if len(selected_columns) > 4:
        return False, "Please select at most 4 columns."
    
    return True, ""

def get_column_sample_data(df: pd.DataFrame, columns: List[str], max_rows: int = 20) -> str:
    """Get sample data for selected columns."""
    sample_df = df[columns].head(max_rows)
    return sample_df.to_string(index=False)


def create_categorization_chain_of_thought_prompt() -> str:
    # Minimal placeholder prompt — replace with your chain-of-thought prompt.
    return (
        "You are an assistant that categorizes CSV columns. Think step-by-step about the values "
        "and return chart types that can model the data after"
        " mapping column names to types (e.g. 'numerical', 'categorical', "
        "'temporal'). Then add strengths and weaknesses for each chart type and pass json vegalite specs for those charts"
    )
 


def categorize_columns_node(state: State, model: Optional[ChatOpenAI] = None) -> State:
    """Node that categorizes selected columns using chain of thought prompting."""
    if model is None:
        raise ValueError("model (ChatOpenAI) is required for categorize_columns_node")
    selected_columns = state.get("selected_columns", [])
    csv_data = state.get("csv_data", "")
    
    chain_of_thought_prompt = create_categorization_chain_of_thought_prompt()
    
    message = HumanMessage(
        content=f"{chain_of_thought_prompt}\n\nColumns: {json.dumps(selected_columns)}\n"
                f"Data sample:\n{csv_data}\n\n"
                f"Return ONLY a JSON object with the categorization for each column."
    )
    
    response = model.invoke([message])
    # Normalize response_text and response message
    if hasattr(response, "content"):
        response_text = response.content
        ai_msg = response
    elif isinstance(response, list) and len(response) > 0:
        # try first item
        item = response[0]
        response_text = getattr(item, "content", str(item))
        ai_msg = item if hasattr(item, "content") else AIMessage(content=response_text)
    else:
        response_text = str(response)
        ai_msg = AIMessage(content=response_text)
    
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

def create_csv_processing_graph():
    """Create a LangGraph workflow for processing CSV data with column selection."""
    
    # Initialize OpenAI model
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    model = ChatOpenAI(
        model="gpt-5-nano",  # Cheapest model LOL
        api_key=api_key
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