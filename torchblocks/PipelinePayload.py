from torchblocks.TrainingMetrics import TrainingMetrics
from torchblocks.EpochResult import EpochResult

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
		self.input_shape = None