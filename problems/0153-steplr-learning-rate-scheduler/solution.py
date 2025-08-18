class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        self.initial_lr=initial_lr
        self.step_size=step_size
        self.gamma=gamma

    

    def get_lr(self, epoch):
        div=epoch//self.step_size

        if div==0:
            return round(self.initial_lr,4)
        else:
            val=self.initial_lr*(self.gamma**div)
            return round(val,4)
            
