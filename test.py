from causalfm.evaluation import compute_pehe, compute_ate_error, evaluate_model
from causalfm.models import StandardCATEModel, IVModel, FrontdoorModel
import torch

# Standard CATE Model
model = StandardCATEModel.from_pretrained("checkpoints/checkpoints_standard/best_model.pth")

# Prepare data
x_train = torch.randn(800, 10)  # Training covariates
a_train = torch.randint(0, 2, (800,1)).float()  # Training treatments
y_train = torch.randn(800)  # Training outcomes
x_test = torch.randn(200, 10)  # Test covariates

# Estimate CATE
result = model.estimate_cate(x_train, a_train, y_train, x_test)
cate_predictions = result['cate']  # Shape: (200,)
# Compute individual metrics
pehe = compute_pehe(cate_predictions, true_ite)
ate_error = compute_ate_error(cate_predictions, true_ite)

# Evaluate on a dataset
from causalfm.models import StandardCATEModel
model = StandardCATEModel.from_pretrained("checkpoints/standard/best_model.pth")
result = evaluate_model(model, "data_test/standard/*_1.csv")
print(f"PEHE: {result.pehe:.4f}, ATE Error: {result.ate_error:.4f}")

# Evaluate on multiple datasets
from causalfm.evaluation.metrics import evaluate_multiple_datasets
results_df = evaluate_multiple_datasets(
    model,
    data_dir="data_test/standard",
    file_pattern="*.csv"
)
print(results_df)