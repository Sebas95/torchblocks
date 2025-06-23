class EpochResult:
	def __init__(self, train_loss, val_loss, val_true_labels, val_predicted_labels, train_true_labels, train_predicted_labels):
		self.train_loss = train_loss
		self.val_loss = val_loss
		self.val_true_labels = val_true_labels
		self.val_predicted_labels = val_predicted_labels
		self.train_true_labels = train_true_labels
		self.train_predicted_labels = train_predicted_labels
