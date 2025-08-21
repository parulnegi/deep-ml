from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    minloss=float("inf")
    epochmin=0
    count=0
    for epoch, loss in enumerate(val_losses):
        if minloss>loss+min_delta:
            minloss=loss
            epochmin=epoch
            count=0
        else:
            count+=1
        if count==patience:
            return (epoch,epochmin)

    return(len(val_losses)-1,len(val_losses)-1)
