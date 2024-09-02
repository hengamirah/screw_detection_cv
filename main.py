YAML
name: Install Dependencies

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python environment
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'   
  # Replace with your desired Python version
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt   
