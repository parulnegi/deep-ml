import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
	"""
	Detect and remove outliers using the IQR method.
	
	Args:
		data: List of numerical values
		k: IQR multiplier for determining outlier bounds (default 1.5)
	
	Returns:
		Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
	"""
	data = np.array(data)
	q1 = np.percentile(data, 25)
	q3 = np.percentile(data, 75)
	Iqr = q3 - q1
	lower_bound =  q1 - (k * Iqr)
	upper_bound = q3 + (k * Iqr)
	mask = (data<lower_bound) | (data>upper_bound)
	cleaned_data = data[~mask]
	outlier= np.where(mask)[0]


	return {
		'cleaned_data': cleaned_data.tolist(),
		 'outlier_indices': outlier.tolist(),
		  'lower_bound': lower_bound.tolist(),
		   'upper_bound': upper_bound.tolist()
	}
	
