"""
MercuryViz Evaluation Script
Tests fidelity (valid specs) and appropriateness (correct chart types)

Author: Sujal Thakkar
Course: CS7295
"""

import json
import os
import sys
import shutil
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Add the current directory to path so we can import the backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the backend function directly instead of using subprocess
from csv_llm_backend import process_csv_with_column_selection

# ============================================================
# OUTPUT FOLDER SETUP
# ============================================================

OUTPUT_FOLDER = "test_results"

def setup_output_folder():
    """Create test_results folder, clean if exists."""
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created output folder: {OUTPUT_FOLDER}/")

# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [
    # ============ sample_data.csv tests ============
    {
        "name": "Test 1: Two numerical (age, salary)",
        "csv": "sample_data.csv",
        "columns": ["age", "salary"],
        "expected_chart_types": ["scatter", "line", "histogram", "bar"]
    },
    {
        "name": "Test 2: Categorical + Numerical (city, salary)",
        "csv": "sample_data.csv",
        "columns": ["city", "salary"],
        "expected_chart_types": ["bar", "box"]
    },
    {
        "name": "Test 3: All three (age, salary, city)",
        "csv": "sample_data.csv",
        "columns": ["age", "salary", "city"],
        "expected_chart_types": ["scatter", "bar", "box", "histogram"]
    },
    
    # ============ large_sample_data.csv tests ============
    {
        "name": "Test 4: Categorical + Numerical (Institution, TOTAL)",
        "csv": "large_sample_data.csv",
        "columns": ["Institution", "TOTAL"],
        "expected_chart_types": ["bar", "box"]
    },
    {
        "name": "Test 5: Two numerical (Total_FEMALE, Total_MALE)",
        "csv": "large_sample_data.csv",
        "columns": ["Total_FEMALE", "Total_MALE"],
        "expected_chart_types": ["scatter", "line", "bar", "histogram"]
    },
    {
        "name": "Test 6: Cat + Cat + Num (Institution, Course, TOTAL)",
        "csv": "large_sample_data.csv",
        "columns": ["Institution", "Course", "TOTAL"],
        "expected_chart_types": ["bar", "heatmap", "box"]
    },
    {
        "name": "Test 7: Multiple numerical (Total_FEMALE, Total_MALE, TOTAL)",
        "csv": "large_sample_data.csv",
        "columns": ["Total_FEMALE", "Total_MALE", "TOTAL"],
        "expected_chart_types": ["scatter", "line", "bar", "histogram"]
    },
]

# ============================================================
# FIDELITY TEST: Check if Vega-Lite spec is valid
# ============================================================

def test_spec_validity(vega_spec):
    """
    Check if a Vega-Lite spec has required fields.
    """
    results = {
        "has_schema": False,
        "has_mark": False,
        "has_encoding": False,
        "has_data": False,
        "is_valid_json": False,
        "passed": False,
        "error": None
    }
    
    try:
        if isinstance(vega_spec, str):
            spec = json.loads(vega_spec)
        else:
            spec = vega_spec
        
        results["is_valid_json"] = True
        results["has_schema"] = "$schema" in spec
        results["has_mark"] = "mark" in spec
        results["has_encoding"] = "encoding" in spec
        results["has_data"] = "data" in spec
        
        results["passed"] = results["has_mark"] and results["has_encoding"]
        
    except json.JSONDecodeError as e:
        results["error"] = f"JSON parse error: {str(e)}"
    except Exception as e:
        results["error"] = str(e)
    
    return results

# ============================================================
# APPROPRIATENESS TEST: Check if chart type makes sense
# ============================================================

def test_chart_appropriateness(chart_name, expected_types):
    """
    Check if the recommended chart type is appropriate.
    """
    if not chart_name:
        return {"passed": False, "matched": None, "got": "Unknown"}
    
    chart_lower = chart_name.lower()
    
    for expected in expected_types:
        if expected.lower() in chart_lower:
            return {"passed": True, "matched": expected, "got": chart_name}
    
    return {"passed": False, "matched": None, "got": chart_name}

# ============================================================
# RUN BACKEND DIRECTLY (not subprocess)
# ============================================================

def run_backend(csv_file, columns, output_prefix):
    """
    Run the backend directly by importing the function.
    """
    # Save output files inside test_results folder
    output_file = os.path.join(OUTPUT_FOLDER, f"{output_prefix}.txt")
    
    print(f"  Calling backend with: {csv_file}, columns={columns}")
    
    try:
        # Call the backend function directly
        result = process_csv_with_column_selection(
            csv_path=csv_file,
            selected_columns=columns,
            output_file=output_file
        )
        
        if result is None:
            print("  Backend returned None")
            return None
        
        # The result contains chart_specs directly
        chart_specs = result.get("chart_specs", {})
        return chart_specs
            
    except Exception as e:
        print(f"  Backend Error: {str(e)}")
        return None

# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation():
    print("=" * 60)
    print("MERCURYVIZ EVALUATION SCRIPT")
    print("=" * 60)
    print(f"Running {len(TEST_CASES)} test cases...")
    print("This may take several minutes due to LLM API calls.\n")
    
    # Setup output folder
    setup_output_folder()
    
    all_results = []
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n[Test {i+1}/{len(TEST_CASES)}] {test['name']}")
        print("-" * 50)
        print(f"  CSV: {test['csv']}")
        print(f"  Columns: {', '.join(test['columns'])}")
        
        # Run the backend
        chart_specs = run_backend(
            test["csv"], 
            test["columns"],
            f"eval_test_{i+1}"
        )
        
        if chart_specs is None:
            print("  Status: SKIPPED (backend failed)")
            all_results.append({
                "test": test["name"],
                "status": "SKIPPED",
                "csv": test["csv"],
                "columns": test["columns"]
            })
            continue
        
        # Get chart types from response
        chart_types = chart_specs.get("chart_types", [])
        print(f"  Charts generated: {len(chart_types)}")
        
        test_result = {
            "test": test["name"],
            "status": "COMPLETED",
            "csv": test["csv"],
            "columns": test["columns"],
            "num_charts": len(chart_types),
            "charts": [],
            "fidelity_pass": 0,
            "fidelity_total": 0,
            "appropriateness_pass": 0,
            "appropriateness_total": 0
        }
        
        # Evaluate each chart
        for j, chart in enumerate(chart_types):
            chart_name = chart.get("name", "Unknown")
            vega_spec = chart.get("vega_lite_spec", {})
            
            # Fidelity test
            fidelity = test_spec_validity(vega_spec)
            test_result["fidelity_total"] += 1
            if fidelity["passed"]:
                test_result["fidelity_pass"] += 1
            
            # Appropriateness test
            appropriateness = test_chart_appropriateness(
                chart_name, 
                test["expected_chart_types"]
            )
            test_result["appropriateness_total"] += 1
            if appropriateness["passed"]:
                test_result["appropriateness_pass"] += 1
            
            # Display results
            fidelity_icon = "PASS" if fidelity["passed"] else "FAIL"
            approp_icon = "PASS" if appropriateness["passed"] else "FAIL"
            
            print(f"    [{j+1}] {chart_name}")
            print(f"        Fidelity: {fidelity_icon}  |  Appropriateness: {approp_icon}")
            
            # Store chart result
            test_result["charts"].append({
                "name": chart_name,
                "fidelity": fidelity,
                "appropriateness": appropriateness
            })
        
        all_results.append(test_result)
    
    # ============================================================
    # CALCULATE SUMMARY
    # ============================================================
    
    total_fidelity_pass = 0
    total_fidelity_tests = 0
    total_approp_pass = 0
    total_approp_tests = 0
    completed_tests = 0
    skipped_tests = 0
    
    for result in all_results:
        if result.get("status") == "SKIPPED":
            skipped_tests += 1
            continue
        completed_tests += 1
        total_fidelity_pass += result.get("fidelity_pass", 0)
        total_fidelity_tests += result.get("fidelity_total", 0)
        total_approp_pass += result.get("appropriateness_pass", 0)
        total_approp_tests += result.get("appropriateness_total", 0)
    
    # Calculate rates
    fidelity_rate = (total_fidelity_pass / total_fidelity_tests * 100) if total_fidelity_tests > 0 else 0
    approp_rate = (total_approp_pass / total_approp_tests * 100) if total_approp_tests > 0 else 0
    overall_score = (fidelity_rate * 0.4 + approp_rate * 0.6) if (total_fidelity_tests > 0 and total_approp_tests > 0) else 0
    
    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nTest Execution:")
    print(f"  Completed: {completed_tests}/{len(TEST_CASES)}")
    print(f"  Skipped:   {skipped_tests}/{len(TEST_CASES)}")
    
    print(f"\nFidelity (Valid Vega-Lite Specs):")
    print(f"  Passed: {total_fidelity_pass}/{total_fidelity_tests}")
    print(f"  Rate:   {fidelity_rate:.1f}%")
    
    print(f"\nAppropriateness (Correct Chart Types):")
    print(f"  Passed: {total_approp_pass}/{total_approp_tests}")
    print(f"  Rate:   {approp_rate:.1f}%")
    
    print(f"\nOverall Score (40% fidelity + 60% appropriateness):")
    print(f"  {overall_score:.1f}%")
    
    # ============================================================
    # SAVE RESULTS TO test_results FOLDER
    # ============================================================
    
    summary = {
        "total_tests": len(TEST_CASES),
        "completed_tests": completed_tests,
        "skipped_tests": skipped_tests,
        "fidelity": {
            "passed": total_fidelity_pass,
            "total": total_fidelity_tests,
            "rate": round(fidelity_rate, 1)
        },
        "appropriateness": {
            "passed": total_approp_pass,
            "total": total_approp_tests,
            "rate": round(approp_rate, 1)
        },
        "overall_score": round(overall_score, 1),
        "timestamp": datetime.now().isoformat()
    }
    
    output_data = {
        "summary": summary,
        "tests": all_results
    }
    
    # Save main results file
    results_file = os.path.join(OUTPUT_FOLDER, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Save summary as separate file for quick reference
    summary_file = os.path.join(OUTPUT_FOLDER, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Move any generated chart JSON files to test_results folder
    for f in os.listdir('.'):
        if f.startswith('eval_test_') and f.endswith('.json'):
            src = f
            dst = os.path.join(OUTPUT_FOLDER, f)
            shutil.move(src, dst)
    
    print(f"\n" + "=" * 60)
    print(f"All results saved to: {OUTPUT_FOLDER}/")
    print(f"  - evaluation_results.json (detailed)")
    print(f"  - summary.json (quick reference)")
    print(f"  - eval_test_*.txt (individual test outputs)")
    print(f"  - eval_test_*_charts.json (generated specs)")
    print("=" * 60)
    
    return summary

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_evaluation()