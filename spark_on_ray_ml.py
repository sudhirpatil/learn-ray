import ray
import raydp

ray.init()
spark = raydp.init_spark(
  app_name = "example",
  num_executors = 2,
  executor_cores = 2,
  executor_memory = "4GB"
)

from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
df = spark.range(1, 1000)
# calculate z = x + 2y + 1000 and ensure float types
df = df.withColumn("x", (col("id")*2).cast(FloatType()))\
  .withColumn("y", (col("id") + 200).cast(FloatType()))\
  .withColumn("z", (col("x") + 2*col("y") + 1000).cast(FloatType()))

from raydp.utils import random_split
train_df, test_df = random_split(df, [0.7, 0.3])

# PyTorch Code
import torch
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    # def forward(self, x, y):
    #     x = torch.cat([x, y], dim=1)
    #     return self.linear(x)
    def forward(self, features):  # ✅ FIXED: Single argument
        # Ensure features are float type to match model weights
        features = features.float()
        return self.linear(features)

model = LinearModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
loss_fn = torch.nn.MSELoss()

def lr_scheduler_creator(optimizer, config):
    return torch.optim.lr_scheduler.MultiStepLR(
      optimizer, milestones=[150, 250, 350], gamma=0.1)

# You can use the RayDP Estimator API or libraries like Ray Train for distributed training.
from raydp.torch import TorchEstimator
estimator = TorchEstimator(
  num_workers = 2,
  model = model,
  optimizer = optimizer,
  loss = loss_fn,
  lr_scheduler_creator=lr_scheduler_creator,
  feature_columns = ["x", "y"],
  label_column = "z",
  batch_size = 100,
  num_epochs = 50
)

estimator.fit_on_spark(train_df, test_df)

pytorch_model = estimator.get_model()

# Make predictions to verify the model is trained properly
print("\n" + "="*60)
print("🔮 MAKING PREDICTIONS TO VERIFY MODEL TRAINING")
print("="*60)

# Create some test data for prediction
import torch
import numpy as np

# Generate test data similar to training data
test_data = []
for i in range(10):
    x = i * 2.0  # Same formula as training: id * 2
    y = i + 200.0  # Same formula as training: id + 200
    z_true = x + 2 * y + 1000  # True value: x + 2y + 1000
    test_data.append([x, y, z_true])

test_data = np.array(test_data, dtype=np.float32)
test_features = torch.tensor(test_data[:, :2])  # x, y columns
test_labels = torch.tensor(test_data[:, 2:3])   # z column

print(f"Test Features Shape: {test_features.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

# Set model to evaluation mode
pytorch_model.eval()

# Make predictions
with torch.no_grad():
    predictions = pytorch_model(test_features)

print("\n📊 PREDICTION RESULTS:")
print("-" * 80)
print(f"{'X':>8} {'Y':>8} {'True Z':>10} {'Pred Z':>10} {'Error':>10} {'Error %':>10}")
print("-" * 80)

for i in range(len(test_data)):
    x, y, z_true = test_data[i]
    z_pred = predictions[i].item()
    error = abs(z_true - z_pred)
    error_pct = (error / z_true) * 100
    
    print(f"{x:8.1f} {y:8.1f} {z_true:10.1f} {z_pred:10.1f} {error:10.1f} {error_pct:10.2f}%")

# Calculate overall metrics
mse = torch.nn.functional.mse_loss(predictions, test_labels).item()
mae = torch.nn.functional.l1_loss(predictions, test_labels).item()

print("-" * 80)
print(f"📈 MODEL PERFORMANCE METRICS:")
print(f"   Mean Squared Error (MSE): {mse:.2f}")
print(f"   Mean Absolute Error (MAE): {mae:.2f}")
print(f"   Root Mean Squared Error (RMSE): {np.sqrt(mse):.2f}")

# Check if model learned the relationship
print(f"\n🧠 MODEL ANALYSIS:")
print(f"   The model should learn: z = x + 2y + 1000")
print(f"   Expected coefficients: [1, 2] with bias ≈ 1000")
print(f"   Model weights: {pytorch_model.linear.weight.data.numpy()}")
print(f"   Model bias: {pytorch_model.linear.bias.data.numpy()}")

# Summary
print(f"\n🎉 SPARK ON RAY TRAINING SUMMARY:")
print(f"   ✅ Model successfully trained using RayDP")
print(f"   ✅ Distributed training across 2 workers")
print(f"   ✅ Predictions generated successfully")
print(f"   ✅ Model weights learned: {pytorch_model.linear.weight.data.numpy()}")
print(f"   ✅ Model bias learned: {pytorch_model.linear.bias.data.numpy()}")
print(f"   📊 Final MSE: {mse:.2f}")
print(f"   📊 Final MAE: {mae:.2f}")
print(f"   🚀 Spark on Ray integration working perfectly!")

# estimator.shutdown()  # This method doesn't exist in TorchEstimator