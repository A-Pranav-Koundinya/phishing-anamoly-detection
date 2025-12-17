# src/consumer.py
import json
from kafka import KafkaConsumer
import requests

# Connect to Kafka (assumes running locally or in docker)
consumer = KafkaConsumer(
    'network-traffic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Listening for network traffic...")

for message in consumer:
    traffic_data = message.value
    
    # Send to our own API for prediction
    try:
        response = requests.post("http://localhost:8000/predict", json=traffic_data)
        result = response.json()
        
        print(f"Packet ID: {traffic_data.get('id', 'N/A')} | "
              f"Pred: {result['prediction']} | "
              f"Error: {result['reconstruction_error']:.4f}")
              
        # In a real system, you would produce this result to a 'alerts' topic here
    except Exception as e:
        print(f"Error processing message: {e}")