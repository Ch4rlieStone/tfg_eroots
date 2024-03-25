from scipy.optimize import rosen
XX=[2,1]
g = pow(XX[0]-1,3)-XX[1] + 1
h = XX[0]+XX[1]-2
if ((g > 0) or (h >0)):
	y= rosen(XX) + 1e6
else:
	y= rosen(XX)
print(y)