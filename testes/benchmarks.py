import numpy as np

def sphere(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-5 <= xi <= 5
	'''
	x = np.array(X)
	return np.sum(x**2)

def quadric(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-5 <= xi <= 5
	'''
	x = np.array(X)
	return np.sum([np.sum(x[:i])**2 for i in range(1,1+len(x))])
	
def rastrigin(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-5.12 <= xi <= 5.12
	'''
	x = np.array(X)
	omega = 2.*np.pi
	return np.sum([i**2 + 10. - 10.*np.cos(omega*i) for i in x])
	
def rosenbrock(X):
	'''
		Minimum:
			f(1,...,1) = 0
		Search Domain:
			-5 <= xi <= 10
	'''
	x = np.array(X)
	return np.sum(100.*(x[1:] - x[:-1]**2)**2 + (1. - x[:-1])**2)
	
def schwefel(X):
	'''
		Minimum:
			f(420.9687,...,420.9687) = 0
		Search Domain:
			-500 <= xi <= 500
	'''
	x = np.array(X)
	n = len(x)
	return 418.9829*n - np.sum([i*np.sin(np.sqrt(np.abs(i))) for i in x])
	
def michalewicz(X,m=10):
	'''
		Minimum:
			n = 2: f(2.20, 1.57) = -1.8013
			n = 5: f(x*) = -4.687658
			n = 10: f(x*) = -9.66015
		Search Domain:
			0 <= xi <= pi
	'''
	x = np.array(X)
	n, s = len(x), 0
	for i in range(n):
		s += np.sin(x[i]) * (np.sin((i + 1) * x[i] / np.pi)**(2. * m))
	return -s
	
def ackley(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-32 <= xi <= 32
	'''
	x = np.array(X)
	a, b, c, d = 20, .2, 2*np.pi, len(x)
	ssq = np.sum(x**2)
	ssc = np.sum(np.cos(c*x))
	t1 = -a*np.exp(-b*np.sqrt(ssq/d))
	t2 = -np.exp(ssc/d)
	return t1+t2+a+np.e
	
def griewank(X,fr=4000):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-600 <= xi <= 600
	'''
	x = np.array(X)
	s, p, n = np.sum(x**2), 1, len(x)
	for j in range(n):
		p *= np.cos(x[j]/np.sqrt(j+1))
	return s/fr-p+1	
	
	
def salomon(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-100 <= xi <= 100
	'''
	x = np.array(X)
	sx = np.sqrt(np.sum(x**2))
	return 1 - np.cos(2*np.pi * sx) + (0.1 * sx)
	
def xin_she_yang(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-5 <= xi <= 5
	'''
	x = np.array(X)
	s = 0
	for i in range(len(x)):
		s += np.abs(x[i])**(i+1) * np.random.uniform(0,1)
	return s

def happy_cat(X,alpha = 0.125):
	'''
		Minimum:
			f(-1,...,-1) = 0
		Search Domain:
			-2 <= xi <= 2
	'''
	x = np.array(X)
	n = len(x)
	s = np.sum(x**2)
	return ((s - n)**2)**alpha + (1./n) * (0.5*s + np.sum(x)) + 0.5
	
def qing(X):
	'''
		Minima:
			f(+-sqrt(i),...,-1) = 0
			f(+-1,+-sqrt(2),...,+-sqrt(n)) = 0
		Search Domain:
			-500 <= xi <= 500
	'''
	x = np.array(X)
	s = 0
	for i in range(len(x)):
		s += (x[i]**2 - (i+1))**2
	return s

def giunta(X):	
	'''
		Minimum:
			n = 2: f(0.4673200277395354,0.4673200169591304) = 0.06447042053690566
		Search Domain:
			-1 <= xi <= 1
	'''
	x = np.array(X)
	s = 0.6
	for i in range(len(x)):
		s += np.sin(1-16*x[i]/15)**2
		s -= np.sin(4-64*x[i]/15)/50
		s -= np.sin(1-16*x[i]/15)
	return s

def brown(X):
	'''
		Minimum:
			f(0,...,0) = 0
		Search Domain:
			-1 <= xi <= 4
	'''
	x = np.array(X)
	s = 0
	for i in range(len(x)-1):
		s += (x[i]**2)**(1+x[i+1]**2)
		s += (x[i+1]**2)**(1+x[i]**2)
	return s
	
def pathological(X):
	'''
		Minimum:
			f(0,...,0) = -1.99600798403
			n = 2: f(0,0) = -1.00
		Search Domain:
			-100 <= xi <= 100
	'''
	x = np.array(X)
	s = 0
	for i in range(len(x)-1):
		n = -0.5 + np.sin(np.sqrt(100.*x[i+1]**2 + x[i]**2))**2
		d = .001*(x[i] - x[i+1])**4 + .5
		s += (n/d)
	return s

'''
	2D benchmarks
'''
def easom(X):
	'''
		Minimum:
			f(pi,pi) = -1
		Search Domain:
			-100 <= x,y <= 100
	'''
	x = np.array(X)
	s = (x[0] - np.pi)**2 + (x[1] - np.pi)**2
	return -np.cos(x[0])*np.cos(x[1])*np.exp(-s)
	

def himmelblau(X):
	'''
		Minima:
			f(3.000000,2.000000)	= 0
			f(-2.805118,3.131312)	= 0
			f(-3.779310,-3.283186)	= 0
			f(3.584428,-1.848126)	= 0
		Search Domain:
			-5 <= x,y <= 5
	'''
	x, y = np.array(X)[:2]
	return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
	
def booth(X):
	'''
		Minimum:
			f(1,3) = 0
		Search Domain:
			-10 <= x,y <= 10
	'''
	x, y = np.array(X)[:2]
	return (x+2*y-7)**2 + (2*x+y-5)**2
	
def beale(X):
	'''
		Minimum:
			f(3.0,0.5) = 0
		Search Domain:
			-4.5 <= x,y <= 4.5
	'''
	x, y = np.array(X)[:2]
	return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
	
def goldstein_price(X):
	'''
		Minimum:
			f(0,-1) = 3
		Search domain:
			-2 <= x,y <= 2
	'''
	x, y = np.array(X)[:2]
	a = (x + y + 1)**2
	b = 19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2
	c = (2*x - 3*y)**2
	d = 18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2
	return (1 + a*b)*(30 + c*d)
	
def bird(X):
	'''
		Minima:
			f(4.70104,3.15294) = -106.764537
			f(-1.58214,-3.13024) = -106.764537
		Search domain:
			-2*pi <= x,y <= 2*pi
	'''
	x, y = np.array(X)[:2]
	cy, sx = np.cos(y), np.sin(x)
	return sx * np.exp((1-cy)**2) + cy * np.exp((1-sx)**2) + (x - y)**2
	
def cross_in_tray(X):
	'''
		Minima:
			f(+-1.349406685353340,+-1.349406608602084) = −2.06261218
		Search Domain:
			-10 <= x,y <= 10
	'''
	x, y = np.array(X)[:2]
	sx, sy, se = np.sin(x), np.sin(y), np.exp(np.abs(100 - np.sqrt(x**2 + y**2)/np.pi))
	return -0.0001 * (1 + np.abs(sx * sy * se))**0.1
	
def dropwave(X):
	'''
		Minimum:
			f(0,0) = -1
		Search Domain:
			-5 <= x,y <= 5
	'''
	x, y = np.array(X)[:2]
	n = 1 + np.cos(12*np.sqrt(x**2 + y**2))
	d = 2 + 0.5 * (x**2 + y**2)
	return -n/d

def eggcrate(X):
	'''
		Minimum:
			f(0,0) = -1
		Search Domain:
			-5 <= x,y <= 5
	'''
	x, y = np.array(X)[:2]
	return x**2 + y**2 + 25*(np.sin(x)**2 + np.sin(y)**2)
	
def sawtoothxy(X):
	'''
		Minimum:
			f(0,0) = 0
		Search Domain:
			-20 <= x,y <= 20
	'''
	x, y = np.array(X)[:2]
	
	r = np.sqrt(x**2 + y**2)
	t = np.arctan2(y,x)
	g = (r**2/(r+1))*(np.sin(r) - np.sin(2*r)/2 + np.sin(3*r)/3 - np.sin(4*r)/4 + 4)
	h = np.cos(2*t-0.5)/2 + np.cos(t) + 2
	return g*h