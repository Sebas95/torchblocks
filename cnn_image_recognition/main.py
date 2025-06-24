import torch
import os
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np

from torchblocks.ClassificationPipeline import ClassificationPipeline
from torchblocks.EpochResult import EpochResult
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Local imports
from conv_net import ConvNet

# Ensure the checkpoints directory exists and split the path
default_model_dir = './checkpoints'
default_model_filename = 'persistedmodel.pth'
default_model_path = os.path.join(default_model_dir, default_model_filename)

class Pipeline(ClassificationPipeline):
	"""
	Concrete implementation of the ClassificationPipeline that defines the custom training loop and test loop.
	"""

	def __init__(self):
		super().__init__()
		print("🚀 Starting CNN Image Recognition Pipeline...")

	def custom_train_loop(self, model, criterion, num_epochs, train_loader, test_loader, optimizer, device):

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
			val_loss, val_true_labels, val_predicted_labels = self.test_loop(model, criterion, test_loader, device)

			# Calculate metrics using accumulated statistics
			epoch_loss = torch.stack(epoch_losses).mean().item()

			# Print validation and training loss
			print(f"Epoch [{epoch+1}/{num_epochs}], "
				  f"Train Loss: {epoch_loss:.4f}, "
				  f"Val Loss: {val_loss:.4f}")

			result[epoch] = EpochResult(
				train_loss=epoch_loss,
				val_loss=val_loss, 
				val_true_labels=val_true_labels,
				val_predicted_labels=val_predicted_labels,
				train_true_labels=train_true_labels,
				train_predicted_labels=train_predicted_labels
			)

		# TODO: Make separate stage to save the model
		print(f"Training model from scratch...")
		if not os.path.isdir(default_model_dir):
			os.makedirs(default_model_dir, exist_ok=True)

		# Save the model
		print(f"Saving model into {default_model_path}...")
		torch.save(model.state_dict(), default_model_path)
		
		return result

	def test_loop(self,model, criterion, test_loader, device):
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

	def load_training_and_test_data(self):

		tensor_transform = transforms.ToTensor()
		normalization_transform = transforms.Normalize((0.5,), (0.5,))
		transform = transforms.Compose([tensor_transform, normalization_transform])

		train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
		test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

		train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

		classes_count = len(train_dataset.classes)

		# Model summary
        # Try to infer input_size from a batch of data
		try:
			batch = next(iter(train_loader))
			if isinstance(batch, (list, tuple)):
				input_tensor = batch[0]
			else:
				input_tensor = batch
			input_size = tuple(input_tensor.shape)

			print(f"Input size inferred from train_loader: {input_size}")
			
		except Exception as e:
			raise RuntimeError("Could not infer input_size from train_loader. Please check your dataset.") from e
		
		return train_loader, test_loader, classes_count, input_size


	def define_training_environment(self):					
		"""Data provisioning phase"""

		# Setup device and data loaders
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		
		# Initialize model and criterion
		model = ConvNet().to(device)

		if args.model_path:
			print(f"Loading model from {args.model_path}...")
			model.load_state_dict(torch.load(args.model_path))

		criterion = nn.CrossEntropyLoss()
		epoch_count = args.epochs
			
		print(f"Using device: {device}")

		# Define the optimizer
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

		return device, model, criterion, epoch_count, optimizer


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train and test CNN")
	
	# Required parameters first
	parser.add_argument("--epochs", type=int, help="Number of epochs", required=True)
	parser.add_argument("--model-path", type=str, help="Path to .pth file with model state_dict")
	args = parser.parse_args()

	# Execute pipeline
	Pipeline().run()

