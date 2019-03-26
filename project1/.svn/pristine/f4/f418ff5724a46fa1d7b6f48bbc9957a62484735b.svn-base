#!/usr/bin/env python

##########################################
##### WRITE YOUR CODE IN THIS FILE #######
##########################################
import numpy as np
import math
import rospy
from sklearn.preprocessing import normalize
from hand_analysis.msg import GraspInfo
from sklearn.decomposition import PCA 
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import csv

class HandAnalysis():
	def __init__(self):
		print("preparison start")
		self.f = rospy.get_param('~train_filename')
		data = np.genfromtxt(fname=self.f, delimiter = ',', skip_header=1)
		glove = data[:, 9:24]
		EMG = data [:, 1:9]
		label = data[:, 0]

		self.trainGlove = SVC(gamma = 'scale').fit(glove, label)

		self.trainEMG = SVC(gamma = 'scale', C = 4.5).fit(normalize(EMG), label)

		self.EMGPreGlove = LinearRegression().fit(EMG, glove)

		self.GloveReduction = PCA(n_components = 2)
		self.GloveReduction.fit(glove)

		self.sub = rospy.Subscriber("/grasp_info", data_class = GraspInfo, callback = self.callback, queue_size = 100)
		self.pub = rospy.Publisher("/labeled_grasp_info", GraspInfo, queue_size = 100)
		print("preparison done")
	def callback(self, msg):
		print("i am in")
		if len(msg.glove) > 0:
			print("one")
			glovepred = np.reshape(msg.glove, (1, -1))
			msg.label = self.trainGlove.predict(glovepred)
			self.pub.publish(msg)
		elif len(msg.emg) > 0:
			print("two")
			EMGpred = np.reshape(msg.emg, (1, -1))
			msg.label = self.trainEMG.predict(normalize(EMGpred))
			msg.glove = self.EMGPreGlove.predict(EMGpred)[0]
			self.pub.publish(msg)
		elif len(msg.glove_low_dim) > 0:
			print("three")
			GloveLow = np.reshape(msg.glove_low_dim, (1, -1))
			msg.glove = self.GloveReduction.inverse_transform(GloveLow)[0]
			self.pub.publish(msg)

if __name__ == '__main__':
	rospy.init_node('analysis', anonymous = True)
	print("hhhhhh")
	cg = HandAnalysis()
	rospy.spin()
