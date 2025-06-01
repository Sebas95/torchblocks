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

default_model_path = './checkpoints/persistedmodel.pth'

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
	@wraps(func)
	def wrapper(*args, **kwargs):
		model, train_losses, val_losses, epoch_number = func(*args, **kwargs)

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

		return model  # Return just the model, clean interface

	return wrapper


@plot_curves
def train_loop(model, criterion, num_epochs):
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
		
	train_losses = []
	epoch_number = []
	val_losses = []
	
	for epoch in range(num_epochs):
		epoch_losses = []
		for x_train, y_train in train_loader:
		
			x_train = x_train.to(device)
			y_train = y_train.to(device)
			
			# Clean up old gradientes
			optimizer.zero_grad()
			
			# Forward pass
			outputs = model(x_train)
		    
			# Compute loss
			loss = criterion(outputs, y_train)
		    
			# Backward pass
			loss.backward()
		    
			# Weigth update
			optimizer.step()
		    
			# Optics collection
			epoch_losses.append(loss)
		
		# Get the lost of last epoch 
		epoch_loss = torch.stack(epoch_losses).mean().item()  
		train_losses.append(epoch_loss)
		epoch_number.append(epoch)
		
		print(f"Epoch {epoch} | Loss: {epoch_loss:.4f}")
		
		# Get the validation loss
		val_loss, _ = test_loop(model, criterion)
		val_losses.append(val_loss)
    
	return model, train_losses, val_losses, epoch_number


def test_loop(model, criterion):
	
	model.eval()  # Set model to evaluation mode
	test_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for x_test, y_test in test_loader:
			x_test, y_test = x_test.to(device), y_test.to(device)
		        
			outputs = model(x_test)
			loss = criterion(outputs, y_test)
			test_loss += loss.item() * x_test.size(0)
		        
			_, predicted = torch.max(outputs, 1)
			total += y_test.size(0)
			correct += (predicted == y_test).sum().item()

		avg_loss = test_loss / total
		accuracy = correct / total

		print(f'Test Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%')
		
		return avg_loss, accuracy


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


