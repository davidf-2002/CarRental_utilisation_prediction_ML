# Car Rental Analysis Dashboard

This project implements a comprehensive analysis dashboard for car rental data, including machine learning predictions.

## Project Structure

```
├── data/               # Data files and data processing scripts
├── src/               # Source code
│   ├── preprocessing/ # Data preprocessing modules
│   ├── analytics/     # Analytics modules
│   └── ml/           # Machine learning modules
├── dashboard/         # Streamlit dashboard files
├── models/           # Saved machine learning models
├── utils/            # Utility functions and helper modules
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks for exploration
├── docs/            # Documentation
└── requirements/    # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements/requirements.txt
```

3. Run the dashboard:
```bash
streamlit run src/app.py
```

## Features

- Interactive data visualization
- Real-time analytics
- Machine learning predictions
- Custom data preprocessing pipeline
- Comprehensive documentation
