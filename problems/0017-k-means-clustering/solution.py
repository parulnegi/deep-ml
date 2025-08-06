def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
	
	for _ in range(max_iterations):
		cluster={c:[] for c in initial_centroids}
		for point in points:
			minval=float("inf")
			closest=None
			for center in initial_centroids:
				dist=sum((p-c)**2 for p,c in zip(point,center))**0.5
				if dist<minval:
					minval=dist
					closest=center
			cluster[closest].append(point)

		new_cent=[]
		for k in cluster.keys():
			if cluster[k]:
				dim=len(cluster[k][0])
				l=len(cluster[k])
				n=[]
				for i in range(dim):
					s=0
					for j in range(l):
						s+=cluster[k][j][i]
					n.append(s/l)
				new_cent.append(tuple(n))
			else:
				new_cent.append(k)
					

		initial_centroids=new_cent	
		

	return initial_centroids