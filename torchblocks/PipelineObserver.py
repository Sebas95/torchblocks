import time

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