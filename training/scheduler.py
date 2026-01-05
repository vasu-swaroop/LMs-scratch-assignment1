from enum import Enum
from jax import numpy as np
from dataclasses import dataclass
class SchedulerType(Enum):
    CONSTANT = 'constant'
    LINEAR = 'linear'
    COSINE_ANNEALING = 'cosine_annealing'
    
@dataclass
class SchedulerPoint():
    step:int
    lr: float

class LRScheduler():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def update(self, step: int):
        pass

class Constant(LRScheduler):
    def __init__(self, point: SchedulerPoint):
        self.lr = point.lr
    def update(self, step: int):
        return self.lr

class Linear(LRScheduler):
    def __init__(self, start: SchedulerPoint, end: SchedulerPoint):
        self.start = start
        self.end = end

    def update(self, step):
        if self.end.step == self.start.step:
            return self.start.lr
        pct = (step - self.start.step) / (self.end.step - self.start.step)
        return self.start.lr + pct * (self.end.lr - self.start.lr)

class CosineAnnealing(LRScheduler):
    def __init__(self, start: SchedulerPoint, end: SchedulerPoint):
        self.start = start
        self.end = end
    
    def update(self, step):
        if self.end.step == self.start.step:
             return self.start.lr
        t = step - self.start.step
        T = self.end.step - self.start.step
        return self.end.lr + 0.5 * (self.start.lr - self.end.lr) * (1 + np.cos(t / T * np.pi))


class ChainSchedulers():
    def __init__(self, scheduler_points:list[SchedulerPoint]):
        self.scheduler_points = scheduler_points
        self.schedulers = []
        
        # Heuristic:
        # All segments are Linear, except:
        # - The second to last interval is CosineAnnealing
        # - The last "interval" (post-last-point) is Constant
        
        # Example 3 points: 0->1 (Linear), 1->2 (Cosine), 2+ (Constant)
        # Example 4 points: 0->1 (Linear), 1->2 (Linear), 2->3 (Cosine), 3+ (Constant)
        
        if len(scheduler_points) < 2:
            raise ValueError("Need at least 2 points for a chain")
            
        num_intervals = len(scheduler_points) - 1
        
        for i in range(num_intervals):
            start = scheduler_points[i]
            end = scheduler_points[i+1]
            
            if i == num_intervals - 1:
                # Last interval -> Cosine
                self.schedulers.append(CosineAnnealing(start, end))
            else:
                # Earlier intervals -> Linear
                self.schedulers.append(Linear(start, end))
                
        # Final constant
        self.schedulers.append(Constant(scheduler_points[-1]))
    
    def update(self, step):
        # Find which interval we are in
        # self.schedulers has len(points) elements (intervals + 1 constant)
        
        for i in range(len(self.scheduler_points) - 1):
            limit_point = self.scheduler_points[i+1]
            if step < limit_point.step:
                return self.schedulers[i].update(step)
                
        # If past all intervals, use the last scheduler (Constant)
        return self.schedulers[-1].update(step)
