import torch
import os 
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import datetime
import argparse
import io

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from functools import wraps
from torchinfo import summary
from torchviz import make_dot

# Local imports
from conv_net import ConvNet
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Ensure the checkpoints directory exists and split the path
default_model_dir = './checkpoints'
default_model_filename = 'persistedmodel.pth'
default_model_path = os.path.join(default_model_dir, default_model_filename)

PLOTS_ROOT_DIR = './plots'

# Map the code name to the class name
VALID_MODELS = {ConvNet.__name__ : "EffConvNet"}


def build_plot_destination_path(model, file_name):

	model_name = type(model).__name__	
	folder_name = VALID_MODELS[model_name]
	
	# Prepare directories for plots
	model_plot_base_directory = f"{PLOTS_ROOT_DIR}/{folder_name}"
	if not os.path.isdir(model_plot_base_directory):
		os.mkdir(model_plot_base_directory)
	
	return f'{PLOTS_ROOT_DIR}/{folder_name}/{file_name}'


def plot_curves(func):
	"""
	A decorator function to plot various performance metrics curves during model training.

	This decorator wraps a function that returns training and validation metrics, and generates
	plots for the following metrics:
	- Learning Curve (Training Loss vs Epoch)
	- Overfitting Plot (Training Loss vs Validation Loss)
	- F1 Score Curve (Training F1 Score vs Validation F1 Score)
	- Accuracy Curve (Training Accuracy vs Validation Accuracy)
	- Precision Curve (Training Precision vs Validation Precision)
	- Recall Curve (Training Recall vs Validation Recall)

	The plots are saved as PNG files in a destination path determined by the `build_plot_destination_path` function.

	Args:
		func (function): A function that returns the following metrics:
			- model: The trained model object.
			- epoch_number: List of epoch numbers.
			- train_losses: List of training loss values.
			- train_f1_scores: List of training F1 scores.
			- train_accuracies: List of training accuracy values.
			- train_precisions: List of training precision values.
			- train_recalls: List of training recall values.
			- val_losses: List of validation loss values.
			- val_f1_scores: List of validation F1 scores.
			- val_accuracies: List of validation accuracy values.
			- val_precisions: List of validation precision values.
			- val_recalls: List of validation recall values.

	Returns:
		function: The wrapped function that generates plots and returns the model object.
	"""
	@wraps(func)
	def wrapper(*args, **kwargs):
		model, epoch_number, train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls, \
		val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls = func(*args, **kwargs)

		# Plot after training
		plt.figure(figsize=(8, 5))
		plt.plot(epoch_number, train_losses, label='Training Loss', marker='o')
		plt.title("Learning Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.grid(True)
		plt.legend()
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'learning_curve.png')
		plt.savefig(file_name)
		plt.close()

		plt.figure(figsize=(10, 5))
		plt.plot(epoch_number, train_losses, label='Train Loss', marker='o')
		plt.plot(epoch_number, val_losses, label='Validation Loss', marker='x')
		plt.title("Overfitting Plot")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'overfit_curve.png')
		plt.savefig(file_name)
		plt.close()

		# Plot F1 scores
		plt.figure(figsize=(10, 5))
		plt.plot(epoch_number, train_f1_scores, label='Train F1 Score', marker='o')
		plt.plot(epoch_number, val_f1_scores, label='Validation F1 Score', marker='x')
		plt.title("F1 Score Curve")
		plt.xlabel("Epoch")
		plt.ylabel("F1 Score")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'f1_score_curve.png')
		plt.savefig(file_name)
		plt.close()

		# Plot accuracies
		plt.figure(figsize=(10, 5))
		plt.plot(epoch_number, train_accuracies, label='Train Accuracy', marker='o')
		plt.plot(epoch_number, val_accuracies, label='Validation Accuracy', marker='x')
		plt.title("Accuracy Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'accuracy_curve.png')
		plt.savefig(file_name)
		plt.close()

			# Plot precisions
		plt.figure(figsize=(10, 5))
		plt.plot(epoch_number, train_precisions, label='Train Precision', marker='o')
		plt.plot(epoch_number, val_precisions, label='Validation Precision', marker='x')
		plt.title("Precision Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Precision")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'precision_curve.png')
		plt.savefig(file_name)
		plt.close()

		# Plot recalls
		plt.figure(figsize=(10, 5))
		plt.plot(epoch_number, train_recalls, label='Train Recall', marker='o')
		plt.plot(epoch_number, val_recalls, label='Validation Recall', marker='x')
		plt.title("Recall Curve")
		plt.xlabel("Epoch")
		plt.ylabel("Recall")
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		file_name = build_plot_destination_path(model, 'recall_curve.png')
		plt.savefig(file_name)
		plt.close()

		return model  # Return just the model, clean interface

	return wrapper


@plot_curves
def train_loop(model, criterion, num_epochs):
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

	epoch_number = []
	
	# Lists to store losses and metrics for each epoch
	train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls = ([] for _ in range(5))

	# Lists to store validation losses and metrics
	val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls = ([] for _ in range(5))
	
	for epoch in range(num_epochs):
		epoch_losses = []
		true_labels = []
		predicted_labels = []
		
		for x_train, y_train in train_loader:
			x_train = x_train.to(device)
			y_train = y_train.to(device)
			
			# Clean up old gradients
			optimizer.zero_grad()
			
			# Forward pass
			outputs = model(x_train)
			
			# Compute loss
			loss = criterion(outputs, y_train)
			
			# Backward pass
			loss.backward()
			
			# Weight update
			optimizer.step()
			
			# Collect optics
			epoch_losses.append(loss)
			
			# Collect true and predicted labels for metrics
			_, predicted = torch.max(outputs, 1)
			true_labels.extend(y_train.cpu().numpy())
			predicted_labels.extend(predicted.cpu().numpy())

		epoch_number.append(epoch)
		
		# Compute metrics for the epoch
		train_loss = torch.stack(epoch_losses).mean().item()  
		train_f1 = f1_score(true_labels, predicted_labels, average='weighted')
		train_accuracy = accuracy_score(true_labels, predicted_labels)
		train_precision = precision_score(true_labels, predicted_labels, average='weighted')
		train_recall = recall_score(true_labels, predicted_labels, average='weighted')

		# Print metrics for the epoch
		print(f"Epoch [{epoch + 1}/{num_epochs}], "
			  f"Train Loss: {train_loss:.4f}, "
			  f"Train F1: {train_f1:.4f}, "
			  f"Train Accuracy: {train_accuracy:.4f}, "
			  f"Train Precision: {train_precision:.4f}, "
			  f"Train Recall: {train_recall:.4f}")

		# Append metrics to their respective lists
		train_losses.append(train_loss)
		train_f1_scores.append(train_f1)
		train_accuracies.append(train_accuracy)
		train_precisions.append(train_precision)
		train_recalls.append(train_recall)
		
		# Get the validation loss
		val_loss, val_accuracy, val_f1, val_precision, val_recall = test_loop(model, criterion)
		
		# Append validation metrics to their respective lists
		val_losses.append(val_loss)
		val_f1_scores.append(val_f1)
		val_accuracies.append(val_accuracy)
		val_precisions.append(val_precision)
		val_recalls.append(val_recall)
    
	return model, epoch_number, train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls, \
		val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls


def test_loop(model, criterion):
	"""
	Evaluates the model on the test dataset and computes the average loss and accuracy.
	Args:
		model (torch.nn.Module): The trained model to be evaluated.
		criterion (torch.nn.Module): The loss function used to compute the loss.
	Returns:
		tuple: A tuple containing:
			- avg_loss (float): The average loss over the test dataset.
			- accuracy (float): The accuracy of the model on the test dataset (between 0 and 1).
	Note:
		- This function assumes that `test_loader` (DataLoader) and `device` (torch.device) are defined in the global scope.
		- The function sets the model to evaluation mode and disables gradient computation for efficiency.
	"""
	
	model.eval()  # Set model to evaluation mode
	test_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		true_labels = []
		predicted_labels = []
		
		for x_test, y_test in test_loader:
			x_test, y_test = x_test.to(device), y_test.to(device)
			
			# Forward pass
			outputs = model(x_test)
			
			# Compute loss
			loss = criterion(outputs, y_test)
			test_loss += loss.item() * x_test.size(0)
			
			# Predictions
			_, predicted = torch.max(outputs, 1)
			total += y_test.size(0)
			correct += (predicted == y_test).sum().item()
			
			# Collect true and predicted labels for metrics
			true_labels.extend(y_test.cpu().numpy())
			predicted_labels.extend(predicted.cpu().numpy())

	# Compute average loss and accuracy
	avg_loss = test_loss / total
	accuracy = correct / total
	f1 = f1_score(true_labels, predicted_labels, average='weighted')
	precision = precision_score(true_labels, predicted_labels, average='weighted')
	recall = recall_score(true_labels, predicted_labels, average='weighted')

	return avg_loss, accuracy, f1, precision, recall


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and test CNN")
	
	# Required parameters first
	parser.add_argument("--epochs", type=int, help="Numer of epochs", required=True)
	parser.add_argument("--model", type=str, help=f"Available model to use: {', '.join(VALID_MODELS.values())}", required=True)
	
	# Optional parameters
	parser.add_argument("--model-path", type=str, help="Path to .pth file with model state_dict")
	args = parser.parse_args()
	
	# Validate the model name
	if args.model not in VALID_MODELS.values():
		raise ValueError(f"❌ Invalid model: '{args.model}'. Valid options are: {', '.join(VALID_MODELS.values())}")

	# Retrieve the data set 
	tensor_transform = transforms.ToTensor()
	normalization_transform = transforms.Normalize((0.5,), (0.5,))
	transform = transforms.Compose([tensor_transform, normalization_transform])

	train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


	# Check if CUDA is available
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
		
	print(f"Using device: {device}")

	# Start the training loop
	model = ConvNet().to(device)
	criterion = nn.CrossEntropyLoss()
	
	# If model passed as parameter, retrieve it from store	
	if args.model_path:
		print(f"Loading model from {args.model_path}...")
		model.load_state_dict(torch.load(args.model_path))
		print(f"Loaded model: {args.model_path}...")
	else:
		print(f"Training model from scratch...")
		if not os.path.isdir(default_model_dir):
			os.makedirs(default_model_dir, exist_ok=True)
		model = train_loop(model, criterion, args.epochs)
		print(f"Saving model into {default_model_path}...")
		torch.save(model.state_dict(), default_model_path)

	# Test loop
	test_loop(model, criterion)

	# Print summary
	input_size = (64, 3, 32, 32)

	# Capture the summary as a string
	bufferStr = io.StringIO()
	summary(model, input_size=input_size)

	# Dummy input
	x = torch.randn(1, 3, 32, 32).to(device)

	# Forward pass
	y = model(x)

	# Create diagram
	file_name = build_plot_destination_path(model, 'model_graph')
	make_dot(y, params=dict(model.named_parameters())).render(file_name, format="png", cleanup=True)


