import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import heapq
import time
from datetime import datetime
from scipy.optimize import curve_fit
from collections import defaultdict
import math
from scipy.stats import truncnorm
from scipy.interpolate import interp1d
from scipy.stats import expon
import threading
import sys
import json
import socket

u = 0 
def counter():
	global u
	while True:
		time.sleep(1)
		u = u  + 1
		print(u)

def func(x, a, b, c):
	return a * np.exp(-b * x) + c
		
class State:
	def __init__(self, name, v):
		self.n = name
		self.v = v
def markov_chain():
	HOST, PORT = "127.0.0.1", 6666
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	random.seed(datetime.now())
	states = defaultdict(dict)
	v = defaultdict(dict)
	v[0][0] ="stop"
	v[0][1] = 0.04
	v[1][0] ="rewind"
	v[1][1] = 0.05
	v[2][0] ="skip"
	v[2][1] = 0.08
	v[3][0] ="pause"
	v[3][1] = 0.1
	v[4][0] ="FF"
	v[4][1] = 0.23
	v[5][0] ="exit"
	v[5][1] =  0.46
	
	
	

	states['start'] = State("start", v)
	
	
	v = defaultdict(dict)
	v[0][0] ="stop"
	v[0][1] = 0.06
	
	v[1][0] ="play"
	v[1][1] = 0.8
	
	states['pause'] = State("pause",v)
	
	v = defaultdict(dict)
	v[0][0] ="skip"
	v[0][1] = 0.03
	v[1][0] ="stop"
	v[1][1] = 0.04
	v[2][0] ="replay"
	v[2][1] =  0.04
	
	v[3][0] ="exit"
	v[3][1] = 0.13
	v[4][0] ="rewind"
	v[4][1] = 0.13
	v[5][0] ="pause"
	v[5][1] = 0.20
	v[6][0] ="FF"
	v[6][1] = 0.41
	
	states['play'] = State("play",v)

	v = defaultdict(dict)
	v[0][0] ="rewind"
	v[0][1] = 0.29
	v[1][0] ="play"
	v[1][1] = 0.58
	states['rewind'] = State("rewind",v)
	
	v = defaultdict(dict)
	v[0][0] ="play"
	v[0][1] = 0.19
	v[1][0] ="exit"
	v[1][1] = 0.81
	states['stop'] = State("stop",v)

	v = defaultdict(dict)
	v[0][0] ="exit"
	v[0][1] = 0.04
	v[1][0] ="FF"
	v[1][1] = 0.25
	v[2][0] ="play"
	v[2][1] = 0.63
	
	states['FF'] = State("FF",v)
	
	v = defaultdict(dict)
	v[0][0] ="exit"
	v[0][1] = 0.03
	v[1][0] ="skip"
	v[1][1] = 0.04
	v[2][0] ="replay"
	v[2][1] = 0.80
	
	
	states['replay'] = State("replay",v)
	
	v = defaultdict(dict)
	v[0][0] ="exit"
	v[0][1] = 0.05
	v[1][0] ="replay"
	v[1][1] = 0.06
	v[2][0] ="skip"
	v[2][1] = 0.83
	states['skip'] = State("skip",v)

	v = defaultdict(dict)
	states['exit'] = State("exit",v)

	
	a = 0
	while(True):
		i = 1
		index = 0
		currentState = states['start']
		#print(currentState)
	
		video_length = 60.0
	
		nextState = currentState
		changed = False
		print(currentState.n)
		video_count  = 1.0
		a = a + 1
		with open("durations.txt", "a") as myfile:
	   			myfile.write("\n" + "Session " + str(a))
		#video_length = random.randint(60,3601)
		print("video length " + str(video_length))
		while (True):
			if(changed == True):
				i = 1
			sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			data = {"Events":{"VoDEvents":{"Name":"Video" + currentState.n, "ServiceIdentifier": "Pulp Fiction", "ServiceInstanceID":2}}}
			j = json.dumps(data)
			print(j)
			try:
			    # Connect to server and send data
			    sock.connect((HOST, PORT))
			    sock.sendall(j.encode())
			    sock.send('\0'.encode())
			    sock.close()

			    # Receive data from the server and shut down
			    
			finally:
			    sock.close()
			if(currentState.n == "exit"):
				break
			index = 0
			x = random.randint(0, 1000)
			#print(x)
			n = currentState.v
			#print(n)
			#		print(x)
			for j in range(len(n)):
					#print(index)
					if (x >= index and x < index + n[j][1] * 1000):
						nextState  = states[str(n[j][0])]
					
						changed = True
					
						
		
						break 
					else:
						index = index + n[j][1]*1000

			duration = sojournTime_SM(currentState.n, n)
			#print(duration[0])
			d = np.power(10.0, duration[0])	
			print("duration: " + str(d))
			with open("durations.txt", "a") as myfile:
	   			myfile.write("\n"+ "state: " + currentState.n + "\n" + "duration :" + str(d) )
			while(float(i/video_length) < d):
				i = i +1
				#print(float(i/video_length))
				time.sleep(1)
				#print(i)
				video_count = video_count + 1
				print("video time:" + str(video_count) + "s")
				if(float(video_count/video_length) >= 1):
					nextState = states["exit"]
					break
			currentState = nextState
			print(currentState.n)
		

def arrivalProcess(number_of_STB, time_range):
	
	x = np.zeros((number_of_STB))
	
	for i in range(number_of_STB):
		x[i] = random.randint(0,time_range-1)
	hist = np.zeros(time_range).astype('float')
	hist, bins = np.histogram(x, bins=time_range)
	nd = np.zeros(time_range).astype('float')
	nd = hist/number_of_STB
	#print (nd)

	f = np.fft.fft(nd)
	print (f)
	a = np.argsort(np.abs(f))[-10:]
	
	
	#print (f)
	for i in range(len(f)):
		if (i not in a):
			f[i] = 0
	print (f)
	#k_keys_sorted = heapq.nlargest(10, f)
	#print (k_keys_sorted)
	inv = np.zeros(len(f))
	inv = np.abs(np.fft.ifft(f))
	

	hist2 = np.zeros(time_range).astype('float')
	x = 0
	while(x < len(hist2)):
		for j in range(0, 10):
			if(x < len(hist2)):
				hist2[x] = inv[j]
				x = x + 1
	
	print (((nd - hist2) ** 2).mean())
	
	plt.figure(1)
	
	

	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	
	plt.bar(center, nd, align='center', width=width)
	width = 0.3 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist2, align='center', width=width, color="red")

	plt.show()

def sojournTime_SM(state, n):
	
	
	#print(hist)
	
	a, b = -10, 0

	x = np.arange(0.00001, 1, 0.00001)
	#print(x)
	x = np.log10(x)
	
	if(state == "start"):
		y = truncnorm.rvs(a, b, scale = 0.4, size=1)
		y2 = truncnorm.cdf(x, a, b, scale = 0.4)
		
	elif(state == "play"):
		y = truncnorm.rvs(a, b, scale = 0.7, size = 1)
		y2 = truncnorm.cdf(x, a, b, scale = 0.7)
	elif(state == "pause"):
		y = truncnorm.rvs(a, b, loc = -1.0, scale = 2, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc = -1.0,  scale = 2)
	elif(state == "stop"):
		y = truncnorm.rvs(a, b, loc = -1.0, scale = 2.5, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc = -1.0, scale = 2.5)
	elif(state == "FF"):
		y = truncnorm.rvs(a, b, loc = -1.5 ,scale = 3, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc =-1.5,  scale = 3)
	elif(state == "replay"):
		y = truncnorm.rvs(a, b, loc = -1.0 , scale = 2, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc = -1.0,	 scale = 2)
	elif(state == "rewind"):
		y = truncnorm.rvs(a, b, loc = -2.5, scale = 3.2, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc = -2.5, scale = 3.2)
	elif(state == "skip"):
		y = truncnorm.rvs(a, b, loc = -1.0, scale = 2.5, size = 1)
		y2 = truncnorm.cdf(x, a, b, loc = -1.0, scale = 2.5)
	elif(state == "exit"):
		sys.exit(0)
		
	
		
	#print(y)
	'''z = np.polyfit(x, y, 10)
	p = np.poly1d(z)
	h = np.arange(0.00001, 1, 0.00001)
	
	h = sum(n[k][1] * p(x)  for k in range(len(n)))
	
	
	a = np.polyfit(x, h, 10)
	
	a = np.poly1d(a)
	print(a(0))
	plt.xlim((-5,0))
	'''
	#plt.plot(x, y2)
	#plt.show()
	return y
	#return p(t)             
	
	#print(y)

	
t = threading.Thread(name='counter', target=counter)
#t.start()
markov_chain()
#sojournTime_SM('a' ,-5)
#arrivalProcess(100000, 100)
