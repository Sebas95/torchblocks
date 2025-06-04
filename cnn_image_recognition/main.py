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
from tqdm import tqdm

# Ensure the checkpoints directory exists and split the path
default_model_dir = './checkpoints'
default_model_filename = 'persistedmodel.pth'
default_model_path = os.path.join(default_model_dir, default_model_filename)

PLOTS_ROOT_DIR = './plots'

# Map the code name to the class name
VALID_MODELS = {ConvNet.__name__ : "EffConvNet"}

class TrainingMetrics:
	def __init__(self, epoch_number, train_losses, train_f1_scores, train_accuracies, train_precisions, 
				 train_recalls, val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls):
		self.epoch_number = epoch_number
		self.train_losses = train_losses 
		self.train_f1_scores = train_f1_scores
		self.train_accuracies = train_accuracies
		self.train_precisions = train_precisions
		self.train_recalls = train_recalls
		self.val_losses = val_losses
		self.val_f1_scores = val_f1_scores 
		self.val_accuracies = val_accuracies
		self.val_precisions = val_precisions
		self.val_recalls = val_recalls

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
		model, metrics = func(*args, **kwargs)
		
		epoch_number = metrics.epoch_number
		train_losses = metrics.train_losses
		train_f1_scores = metrics.train_f1_scores
		train_accuracies = metrics.train_accuracies
		train_precisions = metrics.train_precisions
		train_recalls = metrics.train_recalls
		val_losses = metrics.val_losses
		val_f1_scores = metrics.val_f1_scores
		val_accuracies = metrics.val_accuracies 
		val_precisions = metrics.val_precisions
		val_recalls = metrics.val_recalls

		# Refactored reusable method for plotting
		def plot_metric_curve(metric_name, train_values, val_values=None, ylabel=None):
			plt.figure(figsize=(10, 5))
			plt.plot(epoch_number, train_values, label=f'Train {metric_name}', marker='o')
			if val_values is not None:
				plt.plot(epoch_number, val_values, label=f'Validation {metric_name}', marker='x')
			plt.title(f"{metric_name} Curve")
			plt.xlabel("Epoch")
			plt.ylabel(ylabel or metric_name)
			plt.legend()
			plt.grid(True)
			plt.tight_layout()
			file_name = build_plot_destination_path(model, f'{metric_name.lower().replace(" ", "_")}_curve.png')
			plt.savefig(file_name)
			plt.close()

		# Plot after training
		plot_metric_curve("Learning", train_losses, ylabel="Loss")
		plot_metric_curve("Overfitting", train_losses, val_losses, ylabel="Loss")
		plot_metric_curve("F1 Score", train_f1_scores, val_f1_scores)
		plot_metric_curve("Accuracy", train_accuracies, val_accuracies)
		plot_metric_curve("Precision", train_precisions, val_precisions)
		plot_metric_curve("Recall", train_recalls, val_recalls)

		return model  # Return just the model, clean interface

	return wrapper


@plot_curves
def train_loop(model, criterion, num_epochs, train_loader, test_loader, device):
	"""Train a PyTorch model using the provided data loaders.
	This function handles the training loop for a neural network model, computing various
	metrics for both training and validation sets during training.
	Args:
		model: PyTorch model to be trained
		criterion: Loss function to be used for training
		num_epochs (int): Number of training epochs
		train_loader (DataLoader): DataLoader for training data
		test_loader (DataLoader): DataLoader for validation/test data
		device (torch.device): Device to run the training on (CPU or GPU)
	Returns:
		tuple: Contains:
			- model: Trained PyTorch model
			- TrainingMetrics: Object containing lists of training and validation metrics:
				- epoch_number: List of epoch numbers
				- train_losses: List of training losses per epoch
				- train_f1_scores: List of training F1 scores per epoch
				- train_accuracies: List of training accuracies per epoch
				- train_precisions: List of training precision scores per epoch
				- train_recalls: List of training recall scores per epoch
				- val_losses: List of validation losses per epoch
				- val_f1_scores: List of validation F1 scores per epoch
				- val_accuracies: List of validation accuracies per epoch
				- val_precisions: List of validation precision scores per epoch
				- val_recalls: List of validation recall scores per epoch
	"""
	# Define the optimizer
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

	epoch_number = []
	
	# Lists to store losses and metrics for each epoch
	train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls = ([] for _ in range(5))

	# Lists to store validation losses and metrics
	val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls = ([] for _ in range(5))
	
	for epoch in range(num_epochs):

		# Set the model to training mode
		model.train()

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
		val_loss, val_accuracy, val_f1, val_precision, val_recall = test_loop(model, criterion, test_loader, device)
		
		# Append validation metrics to their respective lists
		val_losses.append(val_loss)
		val_f1_scores.append(val_f1)
		val_accuracies.append(val_accuracy)
		val_precisions.append(val_precision)
		val_recalls.append(val_recall)


	return model, TrainingMetrics(epoch_number, train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls,
					val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls)


def test_loop(model, criterion, test_loader, device):
	"""
	Evaluates the model on the test dataset and computes the average loss and accuracy.
	Args:
		model (torch.nn.Module): The trained model to be evaluated.
		criterion (torch.nn.Module): The loss function used to compute the loss.
		test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
		device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').
	Returns:
		tuple: A tuple containing:
			- avg_loss (float): The average loss over the test dataset.
			- accuracy (float): The accuracy of the model on the test dataset (between 0 and 1).
			- f1 (float): The F1 score of the model on the test dataset.
			- precision (float): The precision of the model on the test dataset.
			- recall (float): The recall of the model on the test dataset.
	Note:
		- This function sets the model to evaluation mode and disables gradient computation for efficiency.
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
	parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
	parser.add_argument("--model", type=str, help=f"Available model to use: {', '.join(VALID_MODELS.values())}", required=True)
	parser.add_argument("--model-path", type=str, help="Path to .pth file with model state_dict")
	args = parser.parse_args()

	def load_data(payload):
		"""Data loading phase"""
		tensor_transform = transforms.ToTensor()
		normalization_transform = transforms.Normalize((0.5,), (0.5,))
		transform = transforms.Compose([tensor_transform, normalization_transform])

		payload.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
		payload.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
		return payload

	def setup_training_environment(payload):
		"""Data provisioning phase"""
		# Validate model name
		if args.model not in VALID_MODELS.values():
			raise ValueError(f"❌ Invalid model: '{args.model}'. Valid options are: {', '.join(VALID_MODELS.values())}")

		# Setup device and data loaders
		payload.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		payload.train_loader = DataLoader(payload.train_dataset, batch_size=64, shuffle=True)
		payload.test_loader = DataLoader(payload.test_dataset, batch_size=64, shuffle=False)
		
		# Initialize model and criterion
		payload.model = ConvNet().to(payload.device)
		payload.criterion = nn.CrossEntropyLoss()
		
		print(f"Using device: {payload.device}")
		return payload

	def execute_training(payload):
		"""Training phase"""
		if args.model_path:
			print(f"Loading model from {args.model_path}...")
			payload.model.load_state_dict(torch.load(args.model_path))
		else:
			print(f"Training model from scratch...")
			if not os.path.isdir(default_model_dir):
				os.makedirs(default_model_dir, exist_ok=True)
			payload.model = train_loop(payload.model, payload.criterion, args.epochs, 
									 payload.train_loader, payload.test_loader, payload.device)
			print(f"Saving model into {default_model_path}...")
			torch.save(payload.model.state_dict(), default_model_path)
		
		return payload

	def create_visualizations(payload):
		"""Visualization phase"""
		# Model summary
		input_size = (64, 3, 32, 32)
		summary(payload.model, input_size=input_size)

		# Model graph
		x = torch.randn(1, 3, 32, 32).to(payload.device)
		y = payload.model(x)
		file_name = build_plot_destination_path(payload.model, 'model_graph')
		make_dot(y, params=dict(payload.model.named_parameters())).render(file_name, format="png", cleanup=True)
		return payload

	# Execute pipeline
	
	class PipelineObserver:
		"""Observer that handles pipeline status updates"""
		def on_start(self, stage): print(f"\n{'='*50}\n🔄 {stage}...\n{'='*50}")
		def on_complete(self, stage): print(f"✅ {stage} complete!\n")
		
	class Pipeline:
		"""Pipeline that executes stages and notifies observers"""
		def __init__(self):
			self.observer = PipelineObserver()
			
		def execute_stage(self, name, func, *args):
			self.observer.on_start(name)
			result = func(*args)
			self.observer.on_complete(name)
			return result
			
		def run(self):
			# Load data
			# Define PipelinePayload class at the beginning
			class PipelinePayload:
				def __init__(self):
					self.train_dataset = None
					self.test_dataset = None
					self.model = None
					self.criterion = None
					self.train_loader = None
					self.test_loader = None
					self.device = None

			# Initialize payload
			payload = PipelinePayload()

			# Load data
			payload = self.execute_stage(
				"Loading data",
				load_data,
				payload
			)
			
			# Setup environment
			payload = self.execute_stage(
				"Setting up training environment",
				setup_training_environment,
				payload
			)
			
			# Execute training
			payload = self.execute_stage(
				"Executing training",
				execute_training,
				payload
			)
			
			# Create visualizations
			self.execute_stage(
				"Creating visualizations",
				create_visualizations,
				payload
			)
			
			print(f"\n{'='*50}")
			print("🎉 Pipeline completed successfully!")
			print(f"{'='*50}")
	
	# Execute pipeline
	Pipeline().run()

