# Madcow Foundations

A Python-based framework for managing and orchestrating AI agent panels with specialized capabilities.

## Overview

The following project is an example supporting the article for MaDCoW (Multi-Agent Dynamic Contribution Workflow) Foundations. It represents a sample implementation of use-case about agents contributing together on a panel discussion. Each panel focuses on specific domains such as philosophy, writing, futuristic concepts, technology, and language models.

## Project Structure
```code
01-madcow-foundations/
├── agents/
│ ├── human.py
│ └── panels/
│   └── *.yaml
├── workflow.py
├── app.py
└── README.md
```

## Getting Started

1. Clone the repository
2. Conda environment setup:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate madcow-foundations
```
4. Update dependencies (optional):
```bash
conda env update -f environment.yml
```
5. Run the application:
```bash
chainlit run app.py
```

## Configuration

Each panel is configured through YAML files located in `agents/panels/`. Modify these files to adjust agent behaviors and capabilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
