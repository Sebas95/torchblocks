import torch
import os 
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from functools import wraps
from torchinfo import summary
from torchviz import make_dot

# Local imports
from conv_net import ConvNet
import time

# Ensure the checkpoints directory exists and split the path
default_model_dir = './checkpoints'
default_model_filename = 'persistedmodel.pth'
default_model_path = os.path.join(default_model_dir, default_model_filename)

PLOTS_ROOT_DIR = './plots'

# Map the code name to the class name
VALID_MODELS = {ConvNet.__name__ : "EffConvNet"}

class TrainingMetrics:
	def __init__(self, epoch_labels, train_losses, train_f1_scores, train_accuracies, train_precisions, 
				 train_recalls, val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls):
		self.epoch_labels = epoch_labels
		
		# Train metrics
		self.train_losses = train_losses 
		self.train_f1_scores = train_f1_scores
		self.train_accuracies = train_accuracies
		self.train_precisions = train_precisions
		self.train_recalls = train_recalls
		
		# Validation metrics
		self.val_losses = val_losses
		self.val_f1_scores = val_f1_scores 
		self.val_accuracies = val_accuracies
		self.val_precisions = val_precisions
		self.val_recalls = val_recalls

class EpochResult:
	def __init__(self, train_loss, val_loss, val_true_labels, val_predicted_labels, train_true_labels, train_predicted_labels):
		self.train_loss = train_loss
		self.val_loss = val_loss
		self.val_true_labels = val_true_labels
		self.val_predicted_labels = val_predicted_labels
		self.train_true_labels = train_true_labels
		self.train_predicted_labels = train_predicted_labels

class PipelinePayload:
	def __init__(self):
		self.model = None
		self.criterion = None
		self.train_loader = None
		self.test_loader = None
		self.device = None
		self.training_metrics: TrainingMetrics = None
		self.epoch_count = None
		self.classes_count = None
		self.training_results: dict[int, EpochResult] = None

class PipelineObserver:
		"""Observer that handles pipeline status updates"""
		def __init__(self):
			self.stage_start_times = {}

		def on_start(self, stage):
			print(f"\n{'='*50}\n🔄 {stage}...\n{'='*50}")
			self.stage_start_times[stage] = time.time()

		def on_complete(self, stage):
			end_time = time.time()
			start_time = self.stage_start_times.get(stage, end_time)
			elapsed = end_time - start_time
			print(f"✅ {stage} complete! (Elapsed: {elapsed:.2f} seconds)\n")

class Pipeline:
	"""Pipeline that executes stages and notifies observers"""
	def __init__(self):
		self.observer = PipelineObserver()
		
	def execute_stage(self, name, func, *args):
		self.observer.on_start(name)
		result = func(*args)
		self.observer.on_complete(name)
		return result
	
	def load_dataset(self, payload):
		"""Data loading phase"""

		# TODO: Make this method abstract so that it can be used with different datasets
		train_loader, test_loader, classes_count = load_training_and_test_data()

		payload.train_loader = train_loader
		payload.test_loader = test_loader
		payload.classes_count = classes_count

		return payload
	

	def build_plot_destination_path(self, model, file_name):

		model_name = type(model).__name__	
		folder_name = VALID_MODELS[model_name]
		
		# Prepare directories for plots
		model_plot_base_directory = f"{PLOTS_ROOT_DIR}/{folder_name}"
		if not os.path.isdir(model_plot_base_directory):
			os.mkdir(model_plot_base_directory)
		
		return f'{PLOTS_ROOT_DIR}/{folder_name}/{file_name}'

	def plot_curves(self, payload):
		""""Plots training and validation metrics curves after training."""

		epoch_labels = payload.training_metrics.epoch_labels

		# Refactored reusable method for plotting
		def plot_metric_curve(metric_name, train_values, val_values=None, ylabel=None):
			plt.figure(figsize=(10, 5))
			plt.plot(epoch_labels, train_values, label=f'Train {metric_name}', marker='o')
			if val_values is not None:
				plt.plot(epoch_labels, val_values, label=f'Validation {metric_name}', marker='x')
			plt.title(f"{metric_name} Curve")
			plt.xlabel("Epoch")
			plt.ylabel(ylabel or metric_name)
			plt.legend()
			plt.grid(True)
			plt.tight_layout()
			file_name = self.build_plot_destination_path(payload.model, f'{metric_name.lower().replace(" ", "_")}_curve.png')
			plt.savefig(file_name)
			plt.close()

		# Plot after training
		plot_metric_curve("Learning", payload.training_metrics.train_losses, ylabel="Loss")
		plot_metric_curve("Overfitting", payload.training_metrics.train_losses, payload.training_metrics.val_losses, ylabel="Loss")
		plot_metric_curve("F1 Score", payload.training_metrics.train_f1_scores, payload.training_metrics.val_f1_scores)
		plot_metric_curve("Accuracy", payload.training_metrics.train_accuracies, payload.training_metrics.val_accuracies)
		plot_metric_curve("Precision", payload.training_metrics.train_precisions, payload.training_metrics.val_precisions)
		plot_metric_curve("Recall", payload.training_metrics.train_recalls, payload.training_metrics.val_recalls)

		return payload
	
	def create_visualizations(self, payload):
		"""Visualization phase"""
		# Model summary
		input_size = (64, 3, 32, 32)
		summary(payload.model, input_size=input_size)

		# Model graph
		x = torch.randn(1, 3, 32, 32).to(payload.device)
		y = payload.model(x)
		file_name = self.build_plot_destination_path(payload.model, 'model_graph')
		make_dot(y, params=dict(payload.model.named_parameters())).render(file_name, format="png", cleanup=True)

		return self.plot_curves(payload)
	
	def compute_metrics_from_payload(self, payload):
		"""
		Compute precision, recall, F1 and accuracy metrics from the payload's confusion matrices.
		Args:
			payload: An object containing:
				- train_confusion_matrix: Dictionary with training confusion matrix statistics
				- validation_confusion_matrix: Dictionary with validation confusion matrix statistics
		Returns:
			TrainingMetrics: An object containing:
				- epoch_labels: List of epoch numbers
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
		def initialize_confusion_matrix(num_epochs, num_classes):
			"""Initialize empty confusion matrix structure for each epoch"""
			confusion_matrix = {}
			metrics = ['tp', 'tn', 'fp', 'fn']
			
			for epoch in range(num_epochs):
				confusion_matrix[epoch] = {
					metric: torch.zeros(num_classes) for metric in metrics
				}
			return confusion_matrix

		# Use helper function to populate confusion matrices
		def populate_confusion_matrices(true_labels, predicted_labels, confusion_matrix, epoch, num_classes):
			"""
			Populates confusion matrix statistics for each class.
			Args:
			true_labels: Array of ground truth labels
			predicted_labels: Array of predicted labels
			confusion_matrix: Dictionary to store confusion matrix stats
			epoch: Current epoch number
			num_classes: Total number of classes
			"""
			for class_idx in range(num_classes):
				confusion_matrix[epoch]['tp'][class_idx] = ((predicted_labels == class_idx) & (true_labels == class_idx)).sum()
				confusion_matrix[epoch]['tn'][class_idx] = ((predicted_labels != class_idx) & (true_labels != class_idx)).sum()
				confusion_matrix[epoch]['fp'][class_idx] = ((predicted_labels == class_idx) & (true_labels != class_idx)).sum()
				confusion_matrix[epoch]['fn'][class_idx] = ((predicted_labels != class_idx) & (true_labels == class_idx)).sum()

		# Populate both matrices using the helper function
		# Initialize confusion matrices for both training and validation
		train_confusion_matrix = initialize_confusion_matrix(payload.epoch_count, payload.classes_count)
		val_confusion_matrix = initialize_confusion_matrix(payload.epoch_count, payload.classes_count)

		val_f1_scores, val_accuracies, val_precisions, val_recalls = ([] for _ in range(4))
		train_f1_scores, train_accuracies, train_precisions, train_recalls = ([] for _ in range(4))

		epoch_labels = []
		train_losses = []
		val_losses = []

		for epoch, epoch_results in payload.training_results.items():

			populate_confusion_matrices(epoch_results.train_true_labels, epoch_results.train_predicted_labels, train_confusion_matrix, epoch, payload.classes_count)
			populate_confusion_matrices(epoch_results.val_true_labels, epoch_results.val_predicted_labels, val_confusion_matrix, epoch, payload.classes_count)

			# Calculate training metrics
			train_precision, train_recall, train_f1, train_accuracy = self.compute_metrics(train_confusion_matrix[epoch])

			# Store training metrics
			train_f1_scores.append(train_f1)
			train_accuracies.append(train_accuracy)
			train_precisions.append(train_precision)
			train_recalls.append(train_recall)

			# Calculate and store validation metrics
			val_precision, val_recall, val_f1, val_accuracy = self.compute_metrics(val_confusion_matrix[epoch])
				
			val_f1_scores.append(val_f1)
			val_accuracies.append(val_accuracy)
			val_precisions.append(val_precision)
			val_recalls.append(val_recall)

			epoch_labels.append(epoch)
			train_losses.append(epoch_results.train_loss)
			val_losses.append(epoch_results.val_loss)

		# Create TrainingMetrics object with all collected metrics
		payload.training_metrics = TrainingMetrics(epoch_labels, train_losses, train_f1_scores, train_accuracies, train_precisions, train_recalls,
						val_losses, val_f1_scores, val_accuracies, val_precisions, val_recalls)
			
		return payload
	

	def setup_training_environment(self, payload):
		"""Data provisioning phase"""
		
		# TODO: Make this method abstract so that it can be used with different datasets
		device, model, criterion, epoch_count = define_training_environment()
		
		# Initialize model and criterion
		payload.device = device
		payload.model = model
		payload.criterion = criterion
		payload.epoch_count = epoch_count

		print(f"Using device: {device}")
		return payload
	
	def compute_metrics(self,confusion_matrix):
		"""
		Compute precision, recall, F1 and accuracy metrics from a confusion matrix.
		Args:
			confusion_matrix (dict): Dictionary containing confusion matrix values
				Expected format:
				{
					'tp': tensor([...]), # True positives per class
					'fp': tensor([...]), # False positives per class
					'fn': tensor([...])  # False negatives per class
				}
				Each tensor should have shape (num_classes,)
		Returns:
			tuple: A tuple containing:
				- precision (float): Mean precision across all classes
				- recall (float): Mean recall across all classes
				- f1 (float): Mean F1 score across all classes
				- accuracy (float): Overall accuracy considering true positives vs total predictions
		Note:
			Small epsilon (1e-7) is added to denominators to avoid division by zero.

		"""

		precisions = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fp'] + 1e-7)
		recalls = confusion_matrix['tp'] / (confusion_matrix['tp'] + confusion_matrix['fn'] + 1e-7)
		f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
				
		precision = precisions.mean().item()
		recall = recalls.mean().item() 
		f1 = f1s.mean().item()
		accuracy = confusion_matrix['tp'].sum() / (confusion_matrix['tp'].sum() + confusion_matrix['fn'].sum())
				
		return precision, recall, f1, accuracy
	

	def train_loop(self, payload):
		"""
		Executes the training loop for a neural network model.
		This function performs the training of a neural network model over a specified number of epochs,
		computing and tracking various performance metrics for both training and validation sets.
		Args:
			payload: An object containing:
				- model: The neural network model to be trained
				- criterion: The loss function
				- epochs: Number of training epochs
				- train_loader: DataLoader for training data
				- test_loader: DataLoader for validation data
				- device: Device to run the training on (CPU/GPU)
		Returns:
			payload: The input payload object updated with:
				- training_metrics: A TrainingMetrics object containing:
					* epoch_labels: List of epoch numbers
					* train_losses: List of training losses per epoch
					* train_f1_scores: List of training F1 scores per epoch
					* train_accuracies: List of training accuracies per epoch
					* train_precisions: List of training precision scores per epoch
					* train_recalls: List of training recall scores per epoch
					* val_losses: List of validation losses per epoch
					* val_f1_scores: List of validation F1 scores per epoch
					* val_accuracies: List of validation accuracies per epoch
					* val_precisions: List of validation precision scores per epoch
					* val_recalls: List of validation recall scores per epoch
				- model: The trained model
		Notes:
			The function uses SGD optimizer with learning rate 0.01 and momentum 0.7.
			Training metrics are printed for each epoch.
		"""
		
		model = payload.model
		
		criterion = payload.criterion
		num_epochs = payload.epoch_count
		train_loader = payload.train_loader
		test_loader = payload.test_loader
		device = payload.device

		# TODO: Make this method abstract so that it can be used with different datasets
		result = custom_train_loop(model, criterion, num_epochs, train_loader, test_loader, device)
		
		# Update payload with training results
		payload.training_results = result

		return payload
	
	def execute_training(self, payload):
		"""Training phase"""
		if args.model_path:
			print(f"Loading model from {args.model_path}...")
			payload.model.load_state_dict(torch.load(args.model_path))
		else:
			print(f"Training model from scratch...")
			if not os.path.isdir(default_model_dir):
				os.makedirs(default_model_dir, exist_ok=True)

			payload = self.train_loop(payload)
			print(f"Saving model into {default_model_path}...")
			torch.save(payload.model.state_dict(), default_model_path)
		
		return payload
				
	def run(self):
		# Load data
		payload = PipelinePayload()

		# Load data
		payload = self.execute_stage(
			"Loading data",
			self.load_dataset,
			payload
		)

		# Setup environment
		payload = self.execute_stage(
			"Setting up training environment",
			self.setup_training_environment,
			payload
		)

		# Execute training
		payload = self.execute_stage(
			"Executing training",
			self.execute_training,
			payload
		)

		payload = self.execute_stage(
			"Executing compute metrics",
			self.compute_metrics_from_payload,
			payload
		)
			
		# Create visualizations
		self.execute_stage(
			"Creating visualizations",
			self.create_visualizations,
			payload
		)
			
		print(f"\n{'='*50}")
		print("🎉 Pipeline completed successfully!")
		print(f"{'='*50}")


def custom_train_loop(model, criterion, num_epochs, train_loader, test_loader, device):

	# Define the optimizer
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

	result = {}

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
			
			# Get binary metrics for each batch
			batch_true = y_train.cpu().numpy()
			batch_pred = predicted.cpu().numpy()

			# Convert predictions and labels to numpy arrays for metrics
			# Append to the lists
			true_labels.extend(batch_true.tolist())
			predicted_labels.extend(batch_pred.tolist())

		# At the end of the loop, convert lists to numpy arrays
		train_true_labels = np.array(true_labels) 
		train_predicted_labels = np.array(predicted_labels)

		# Get validation metrics
		val_loss, val_true_labels, val_predicted_labels = test_loop(model, criterion, test_loader, device)

		# Calculate metrics using accumulated statistics
		epoch_loss = torch.stack(epoch_losses).mean().item()

		result[epoch] = EpochResult(
			train_loss=epoch_loss,
			val_loss=val_loss, 
			val_true_labels=val_true_labels,
			val_predicted_labels=val_predicted_labels,
			train_true_labels=train_true_labels,
			train_predicted_labels=train_predicted_labels
		)
	
	return result

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

		# At the end of the loop, convert lists to numpy arrays
		true_labels = np.array(true_labels) 
		predicted_labels = np.array(predicted_labels)

	# Compute average loss and accuracy
	avg_loss = test_loss / total

	return avg_loss, true_labels, predicted_labels

def load_training_and_test_data():

	tensor_transform = transforms.ToTensor()
	normalization_transform = transforms.Normalize((0.5,), (0.5,))
	transform = transforms.Compose([tensor_transform, normalization_transform])

	train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

	train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

	classes_count = len(train_dataset.classes)
	return train_loader, test_loader, classes_count


def define_training_environment():					
	"""Data provisioning phase"""
	# Validate model name
	if args.model not in VALID_MODELS.values():
		raise ValueError(f"❌ Invalid model: '{args.model}'. Valid options are: {', '.join(VALID_MODELS.values())}")

	# Setup device and data loaders
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	# Initialize model and criterion
	model = ConvNet().to(device)
	criterion = nn.CrossEntropyLoss()
	epoch_count = args.epochs
		
	print(f"Using device: {device}")
	
	return device, model, criterion, epoch_count 


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and test CNN")
	
	# Required parameters first
	parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
	parser.add_argument("--model", type=str, help=f"Available model to use: {', '.join(VALID_MODELS.values())}", required=True)
	parser.add_argument("--model-path", type=str, help="Path to .pth file with model state_dict")
	args = parser.parse_args()

	# Execute pipeline
	Pipeline().run()

