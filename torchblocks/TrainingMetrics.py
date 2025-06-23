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