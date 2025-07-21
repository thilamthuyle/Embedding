import time

# Simulating dynamically changing the model name
model_names = [
    'gte-large-en-v1.5',
    "sentence-camembert-large",
    "sentence-camembert-base"
]

MODEL_NAME_FILE = 'model_name.txt'

for model_name in model_names:
    # Write the model name to a temporary file
    with open(MODEL_NAME_FILE, 'w') as f:
        f.write(model_name)
    print(f"Set model name to {model_name}")
    
    # Simulate waiting time before changing to the next model
    time.sleep(10)  # Wait 10 seconds before changing the model
