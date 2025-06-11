# CogniQuantum Project Structure

This document outlines the directory and file structure of the CogniQuantum project.


```
.
├── .gitignore
├── cli/
│   ├── init.py
│   ├── handler.py
│   └── main.py
├── examples/
│   ├── sample_questions.txt
│   └── v2_demo_script.sh
├── fetch_llm_v2.py
├── installation_guide.md
├── llm_api/
│   ├── init.py
│   ├── cogniquantum/
│   │   ├── init.py
│   │   ├── analyzer.py
│   │   ├── engine.py
│   │   ├── enums.py
│   │   ├── system.py
│   │   └── tracker.py
│   ├── providers/
│   │   ├── init.py
│   │   ├── base.py
│   │   ├── claude.py
│   │   ├── enhanced_claude_v2.py
│   │   ├── enhanced_gemini_v2.py
│   │   ├── enhanced_huggingface_v2.py
│   │   ├── enhanced_ollama_v2.py
│   │   ├── enhanced_openai_v2.py
│   │   ├── gemini.py
│   │   ├── huggingface.py
│   │   ├── ollama.py
│   │   └── openai.py
│   └── utils/
│       ├── init.py
│       ├── helper_functions.py
│       └── performance_monitor.py
├── PROJECT_STRUCTURE.md
├── quick_test_v2.py
├── README.md
├── requirements.txt
├── test_all_v2_providers.py
└── tests/
├── init.py
├── test_cli.py
├── test_cogniquantum.py
└── test_providers.py
```

## Directory and File Overview

### Root Directory

- **`.gitignore`**: Specifies files and directories that Git version control should ignore.
- **`fetch_llm_v2.py`**: A simple entry point that starts the command-line interface. The main logic resides in the `cli/` directory.
- **`installation_guide.md`**: A detailed guide for setting up and installing the project.
- **`PROJECT_STRUCTURE.md`**: This file, outlining the project's structure.
- **`quick_test_v2.py`**: A diagnostic script to quickly verify the user's environment setup (API keys, Ollama connection, etc.).
- **`README.md`**: The main documentation file containing the project's overview, purpose, features, and usage instructions.
- **`requirements.txt`**: Lists the Python libraries and their versions required to run the project.
- **`test_all_v2_providers.py`**: A comprehensive test script that checks the operation of all configured V2 providers in each mode and performs simple performance measurements.

### cli/

This directory contains all the logic for the command-line interface.

- **`__init__.py`**: Allows the cli directory to be treated as a Python package.
- **`handler.py`**: The core logic class of the CLI. It handles request processing, provider fallbacks, and generating suggestions.
- **`main.py`**: The main entry point of the CLI application. It handles argument parsing and orchestrates the overall execution flow.

### examples/

This directory contains sample scripts and data files demonstrating the project's functionality.

- **`sample_questions.txt`**: A list of sample questions for testing each mode: efficient, balanced, and decomposed.
- **`v2_demo_script.sh`**: The latest shell script for trying out various modes and providers using `fetch_llm_v2.py`.

### llm_api/

The main package containing the core reasoning logic and communication with LLM providers.

- **`__init__.py`**: Allows the llm_api directory to be treated as a Python package.

#### cogniquantum/

The core of the CogniQuantum V2 system, broken down by responsibility.

- **`__init__.py`**: Exposes the main `CogniQuantumSystemV2` class to the outside.
- **`analyzer.py`**: Contains the `AdaptiveComplexityAnalyzer` class for analyzing prompt complexity using NLP and keyword-based methods.
- **`engine.py`**: Contains the `EnhancedReasoningEngine` which executes different reasoning strategies (low, medium, high complexity) based on the analysis.
- **`enums.py`**: Defines enumerations used in the system, such as `ComplexityRegime`.
- **`system.py`**: Contains the `CogniQuantumSystemV2` class, which orchestrates the entire problem-solving process.
- **`tracker.py`**: Contains data classes for tracking performance metrics and solutions, such as `ReasoningMetrics` and `SolutionTracker`.

#### providers/

A collection of modules responsible for the actual communication with LLM services.

- **`__init__.py`**: Provides a factory function to dynamically import providers and return the appropriate provider instance upon request.
- **`base.py`**: Defines the abstract base classes (`LLMProvider`, `EnhancedLLMProvider`) that all provider classes inherit from.
- **`openai.py`, `claude.py`, `gemini.py`, `ollama.py`, `huggingface.py`**: Basic provider classes that call the standard APIs for each LLM service.
- **`enhanced_*_v2.py`**: Advanced providers that integrate the functionality of the CogniQuantum system. They wrap the standard providers to execute complexity-adaptive reasoning.

#### utils/

Helper functions and auxiliary classes used throughout the project.

- **`__init__.py`**: Allows the utils directory to be treated as a Python package.
- **`helper_functions.py`**: Provides auxiliary functions, such as reading from files/pipes and formatting JSON.
- **`performance_monitor.py`**: Measures performance metrics like processing time and token usage.

### tests/

This directory contains unit and integration tests for the project.

- **`__init__.py`**: Allows the tests directory to be treated as a Python package.
- **`test_cli.py`**: Tests the command-line arguments and fallback functionality of `fetch_llm_v2.py`.
- **`test_cogniquantum.py`**: Tests the complexity analysis and reasoning logic of the `cogniquantum` module.
- **`test_providers.py`**: Tests the dynamic loading and caching features of the providers.