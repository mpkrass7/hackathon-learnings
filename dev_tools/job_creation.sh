databricks jobs create --profile hershey-sandbox --json '{
  "name": "[hackathon-your_name_here] NYC Taxi Trip Metrics by Pickup Hour",
  "tasks": [{
    "task_key": "run_sample_sql",
    "sql_task": {
      "warehouse_id": "91ec2b975e4ebe0b",
      "file": {
        "path": "replace_me_with_sql_path",
        "source": "WORKSPACE"
      }
    }
  }]
}'
