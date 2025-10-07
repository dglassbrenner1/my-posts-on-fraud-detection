---
layout: custom      # use your main layout
title: 11. Snowflake and dbt demonstation
nav_order: 12
has_toc: true
nav_enabled: true
use_math: true
---

# 11. Snowflake and dbt demonstration

This demonstration builds a fraud detection analytics pipeline using the IEEE Fraud Detection dataset, with data stored and processed in Snowflake. The pipeline includes raw data ingestion into Snowflake schemas, seed data management, staging data checks, and the creation of model-ready datasets with integrated data quality testing and documentation.

## 11.1 Demonstration Overview

We ingest raw transaction and identity data from the IEEE Fraud Detection dataset into Snowflake schemas and manages static seed data for testing. The staging layer performs essential data checks, mainly for nulls and duplicates, to ensure data reliability for downstream analysis and modeling.

### Data used

For this demonstration, I switched up the data and used the IEEE-CIS Fraud Detection dataset instead of the Fraud Handbook data.  The IEEE data is a widely used benchmark dataset for detecting fraudulent online transactions.

The dataset contains detailed transaction and identity information from real-world e-commerce data, with 590,540 transactions labeled as fraudulent or legitimate.

We use two of its files:
- **train_transaction.csv**: Transactional features for training data
- **train_identity.csv**: Identity features linked to transactions for training

The dataset was originally released as part of a Kaggle competition hosted by IEEE and Vesta Corporation:
[IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

To use this dataset, you will need to create a Kaggle account and accept the competition rules before downloading. The project leverages this rich dataset to build effective fraud detection pipelines.

### Key Features of the Demonstration

- Raw data sources documented with lineage and descriptions
- Seed data managed as version-controlled CSV files, with schema and tests
- Staging models built and tested for quality assurance
- Automated refresh of seeds and transformations with dbt commands
- Comprehensive documentation generated with `dbt docs`

## 11.2 Snowflake Integration

This dbt project is integrated with Snowflake, leveraging its cloud data warehouse capabilities to efficiently manage and perform quality checks on large-scale fraud detection data.

- **Data Storage and Schemas**  
  Raw and error-checked data are stored in distinct Snowflake schemas to maintain clear separation and lineage. The RAW schema contains ingested original datasets, while error-checked and lightly validated data is stored in the staging schema for downstream analysis.

- **Connection and Execution**  
  dbt connects to Snowflake using configured profiles containing account information, warehouse, database, and schema targets. This enables fast compilation and execution of SQL transformations directly on Snowflake's compute resources.

- **Seed Loading**  
  Static seed datasets are loaded as Snowflake tables using `dbt seed` into designated schemas. Column data types and loading behavior are controlled via dbt configuration to ensure correct structure and query performance.

- **Role and Permission Management**  
  Appropriate Snowflake roles are used for read access to raw data and write access to transformation outputs. This enforces security and access control within the data platform.

This integration ensures that the fraud detection pipeline is scalable, maintainable, and leverages Snowflake's features for enterprise-grade analytics.


## 11.3 Seed Data

Seeds provide static sample data used for testing, development, and reference. They live in the `seeds/` directory and are loaded with the `dbt seed` command into the designated schema.

Our seed files are:

- `sample_train_transaction.csv`: Sample transactions with features and fraud labels
- `sample_train_identity.csv`: Corresponding identity features for transactions

The seed schemas include column-level descriptions and tests such as uniqueness and non-null constraints to ensure data integrity.

## 11.4 How to Run

1. Load seeds:

    ```
    dbt seed
    ```

2. Run transformations:

    ```
    dbt run
    ```

3. Execute tests:

    ```
    dbt test
    ```

4. Generate and serve documentation:

    ```
    dbt docs generate
    dbt docs serve
    ```

The project is configured to load seeds into the `staging` schema and includes automated testing on critical columns.

## 11.5 Additional Notes

This project forms the foundation for developing fraud detection models and dashboards by ensuring clean, tested input data and transformations. You would normally follow this by integrating feature engineering and model training steps in the pipeline, but I just wanted this to be a simple demonstration project using dbt and Snowflake.

## 11.6 GitHub Repository

The full source code, seed data, dbt models, and configuration for this demonstration are available on GitHub:

[https://github.com/dglassbrenner1/ieee-fraud-detection-dbt](https://github.com/dglassbrenner1/ieee-fraud-detection-dbt)

Visit the repository to review the project structure, clone for your own experiments, or submit issues and pull requests for improvements.

<br>

<table width="100%">
  <tr>
    <td align="left">
      <a href="/10-Hugging-Face-API-deployment.html">← Previous: 10. Hugging Face API deployment</a>
    </td>
    <td align="right">
      Next: Post to come! →</a>
    </td>
  </tr>
</table>

