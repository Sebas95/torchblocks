import torch
import os 
import matplotlib.pyplot as plt

from torchblocks.TrainingMetrics import TrainingMetrics
from torchblocks.PipelinePayload import PipelinePayload
from torchblocks.PipelineObserver import PipelineObserver
from torchblocks.EpochResult import EpochResult
from torch.utils.data import DataLoader
from torchinfo import summary
from torchviz import make_dot
from abc import ABC, abstractmethod
import numpy as np

PLOTS_ROOT_DIR = './plots'

class ClassificationPipeline(ABC):
	"""Pipeline that executes stages and notifies observers"""
	def __init__(self):
		self.observer = PipelineObserver()
		
	def execute_stage(self, name, func, *args):
		self.observer.on_start(name)
		result = func(*args)
		self.observer.on_complete(name)
		return result
	
	def load_dataset(self, payload : PipelinePayload):
		"""Data loading phase"""

		train_loader, test_loader, classes_count, input_size = self.load_training_and_test_data()

		payload.train_loader = train_loader
		payload.test_loader = test_loader
		payload.classes_count = classes_count
		payload.input_size = input_size

		return payload

	def analyze_data(self, payload : PipelinePayload):
		"""Data analysis phase"""
	
		# Analyze class balance in the training set
		class_counts = [0] * payload.classes_count
		for _, labels in payload.train_loader:
			for label in labels:
				class_counts[label] += 1

		max_count = max(class_counts)
		min_count = min(class_counts)
		balance_ratio = min_count / max_count if max_count > 0 else 0

		if balance_ratio < 0.8:
			print(f"⚠️  Unbalanced dataset detected: class counts = {class_counts}")
		else:
			print(f"✅ Balanced dataset: class counts = {class_counts}")

		return payload
	
	@abstractmethod
	def custom_train_loop(self):
		"""		Abstract method for custom training loop.
		Should be implemented in subclasses to provide specific training logic.
		"""

		result: dict[int, EpochResult] = {}
		
		return result

	@abstractmethod
	def test_loop(self):
		"""
		Abstract method for testing loop.
		Should be implemented in subclasses to provide specific testing logic.
		"""
		avg_loss = 0.0
		true_labels = np.array([])
		predicted_labels = np.array([])

		return avg_loss, true_labels, predicted_labels

	@abstractmethod
	def load_training_and_test_data(self):
		"""
		Abstract method to load training and test data.
		Should be implemented in subclasses to provide specific dataset loading logic.
		"""
		train_loader = DataLoader([])
		test_loader = DataLoader([])
		classes_count = 0
		input_size = (0, 0, 0)  # Default input size, should be updated in subclasses

		return train_loader, test_loader, classes_count, input_size

	@abstractmethod
	def define_training_environment(self):
		"""
		Abstract method to define the training environment.
		Should be implemented in subclasses to provide specific environment setup logic.
		"""
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model = None  # Should be defined in subclasses
		criterion = None  # Should be defined in subclasses
		epoch_count = 0  # Should be defined in subclasses
		optimizer = None

		return device, model, criterion, epoch_count, optimizer

	def build_plot_destination_path(self, model, file_name):

		model_name = type(model).__name__	
		folder_name = model_name
		
		# Prepare directories for plots
		model_plot_base_directory = f"{PLOTS_ROOT_DIR}/{folder_name}"
		if not os.path.isdir(model_plot_base_directory):
			os.mkdir(model_plot_base_directory)
		
		return f'{PLOTS_ROOT_DIR}/{folder_name}/{file_name}'

	def plot_curves(self, payload : PipelinePayload):
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
	
	def create_visualizations(self, payload : PipelinePayload):
		"""Visualization phase"""

		summary(payload.model, input_size=payload.input_size)

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
	

	def setup_training_environment(self, payload : PipelinePayload):
		"""Data provisioning phase"""
		
		device, model, criterion, epoch_count, optimizer = self.define_training_environment()
		
		# Initialize model and criterion
		payload.device = device
		payload.model = model
		payload.criterion = criterion
		payload.epoch_count = epoch_count
		payload.optimizer = optimizer

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
	

	def execute_training(self, payload : PipelinePayload):
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
		optimizer = payload.optimizer

		result = self.custom_train_loop(model, criterion, num_epochs, train_loader, test_loader, optimizer, device)

		# Update payload with training results
		payload.training_results = result

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

		payload = self.execute_stage(
			"Analyzing data",
			self.analyze_data,
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
