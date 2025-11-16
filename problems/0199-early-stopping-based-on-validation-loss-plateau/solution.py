def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
	# Your code here
    if not val_losses:
        return []
    best_loss=val_losses[0]
    counter=0
    answer=[False]*len(val_losses)

    for i in range(1, len(val_losses)):
        if val_losses[i]< best_loss-min_delta:
            counter=0
        else:
            counter+=1
        
        best_loss=min(best_loss,val_losses[i])

        if counter==patience:
            answer[i]=True
        # print("/n", i, counter)

    
    return answer


	









