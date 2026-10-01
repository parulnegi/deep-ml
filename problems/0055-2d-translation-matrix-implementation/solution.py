import numpy as np
def translate_object(points, tx, ty):
	p = np.array([tx,ty,1])
	points = np.hstack((points , np.ones((len(points),1))))
	add = np.identity((3))
	add[ :, 2] = p
	trans = add @ points.T
	trans = trans.T[:, :-1]
	return trans


	


