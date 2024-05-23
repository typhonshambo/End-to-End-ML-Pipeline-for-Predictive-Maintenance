import os

def create_project_structure(base_path):
    # Define the folder structure
    folders = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models/saved_model",
        "deployment/k8s",
        "monitoring/prometheus",
        "monitoring/grafana/dashboards",
    ]

    # Define the files to create
    files = {
        "data/simulated_data.csv": "",
        "notebooks/exploratory_data_analysis.ipynb": "",
        "src/__init__.py": "",
        "src/data_preprocessing.py": "",
        "src/feature_engineering.py": "",
        "src/train_model.py": "",
        "src/evaluate_model.py": "",
        "src/pipeline.py": "",
        "models/saved_model/.gitkeep": "",
        "deployment/Dockerfile": "",
        "deployment/k8s/deployment.yaml": "",
        "deployment/k8s/service.yaml": "",
        "deployment/requirements.txt": "",
        "monitoring/prometheus/prometheus.yml": "",
        "monitoring/grafana/dashboards/.gitkeep": "",
        "monitoring/grafana/grafana.ini": "",
        "monitoring/docker-compose.yml": "",
        ".gitignore": "",
        "README.md": "# Predictive Maintenance ML Pipeline\n",
        "setup.py": "",
    }

    # Create the folders
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_path}")

    # Create the files
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        with open(full_path, 'w') as f:
            f.write(content)
        print(f"Created file: {full_path}")

if __name__ == "__main__":
    base_path = "predictive-maintenance"
    create_project_structure(base_path)
    print(f"Project structure created at: {os.path.abspath(base_path)}")
