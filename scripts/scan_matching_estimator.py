import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.scan_matching import ScanMatching

def ScanMatching_estimator(self, estimator):
    self.estimator = estimator