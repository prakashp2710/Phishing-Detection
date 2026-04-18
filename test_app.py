import requests

# Test data with all 14 features (matching trained model)
test_data = {
    'MemoryComplaints': 1,
    'SleepQuality': 3.0,
    'CholesterolHDL': 40.0,
    'BehavioralProblems': 1,
    'FunctionalAssessment': 4.0,
    'ADL': 6.0,
    'MMSE': 22.0,
    'FamilyHistoryAlzheimers': 1,
    'CardiovascularDisease': 1,
    'Diabetes': 1,
    'Hypertension': 1,
    'BMI': 28.0,
    'CholesterolLDL': 150.0,
    'EducationLevel': 2,
}

# Test the Flask app
response = requests.post('http://127.0.0.1:5000/predict', data=test_data)

print('Status Code:', response.status_code)
print('Response Text:', response.text[:200] + '...' if len(response.text) > 200 else response.text)
if 'Error:' in response.text:
    print('❌ Error found in response')
elif '⚠️ Alzheimer' in response.text:
    print('✅ Prediction successful: Alzheimer\'s Disease Detected')
elif '✅ No Alzheimer' in response.text:
    print('✅ Prediction successful: No Alzheimer\'s Disease Detected')
else:
    print('❓ Prediction response unclear')