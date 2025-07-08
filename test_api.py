import requests

url = "http://127.0.0.1:5000/predict"
 
data = {
    "deployment_id": "demo_deploy_001",
    "lines_added": 550,
    "lines_deleted": 250,
    "commit_count": 15,
    "error_rate": 0.07,
    "latency_ms": 450,
    "latency_increase": 0.3,
    "test_coverage": 0.75,
    "bug_count": 6,
    "deploy_type": "minor",
    "env": "prod"
}

try:
    response = requests.post(url, json=data)
    print(" Status Code:", response.status_code)
    print(" JSON Output:\n", response.json())
except Exception as e:
    print("Error occurred:", e)
    print("Raw response:\n", response.text)
