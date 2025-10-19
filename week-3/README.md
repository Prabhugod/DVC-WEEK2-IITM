# Week 3: IRIS Pipeline with Feast Feature Store

This repository demonstrates the integration of **Feast** (Feature Store) into the existing IRIS ML pipeline.  
The objective is to manage features centrally and fetch them for both **training** and **online inference**.

---

## **Folder Structure and File Description:**

```text
week-3
    ├── data
    │   ├── data_feast.parquet
    │   └── iris_data_adapted_for_feast.csv
    ├── iris_feature_feast_repo
    │   └── feature_repo
    │       ├── data
    │       │   ├── driver_stats.parquet
    │       │   ├── online_store.db
    │       │   └── registry.db
    │       ├── example_repo.py
    │       ├── feature_store.yaml
    │       └── test_workflow.py
    ├── model
    │   └── metrics.txt
    ├── README.md
    └── week3_assignment.ipynb
```
---

## **Description of Key Components**

### **data/**
- Contains the **Iris dataset** that has been provided in the Github.
- `iris_data_adapted_for_feast.csv`: Raw CSV dataset containing `iris_id`, `event_timestamp`, and flower measurements for use with Feast.
- `data_feast.parquet`: Optimized Parquet file generated from the CSV for ingestion into the Feast feature store.

### **iris_feature_feast_repo/**
-  Contains the **Feast feature store repository** setup and configurations.
- `feature_repo/feature_store.yaml`: Main Feast configuration file defining registry, provider, and online store settings.
- `feature_repo/example_repo.py`: Defines entity (iris_id), feature views (mapping features from Parquet), and batch source.
- `feature_repo/data/registry.db` & `online_store.db`: Internal databases used by Feast to register and serve features.

### **feature_repo/**
- Contains the **Feast repository** configuration.
- `entities.py`: Defines the entity (`iris_id`) for Feast.
- `feature_views.py`: Defines feature views that map columns from the Parquet dataset to Feast features.

### **models/**
- Stores the trained `model file` and `metric file`.
- `metrics.txt`: Records model evaluation metrics (accuracy, etc.) during training.

### **week3_assignment.ipynb/**
- The primary notebook file demonstrating the complete workflow:
- Dataset preparation (`CSV → Parquet conversion`)
- Feast setup and feature registration
- Historical feature retrieval and model training
- Online feature retrieval and inference demonstration

### **README.md**
- Provides an overview of the project, setup instructions, and utility of each folder/file
