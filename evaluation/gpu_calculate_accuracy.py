from PIL import Image
import os, sys
import numpy
import cv2
from sklearn.metrics import accuracy_score

SHAPE = (192, 608)

def calc_class_accuracy(correct, total):
	"""
	Returns mean class accuracy (float)
	correct: numpy 1d array
		number of correctly classified inputs per class
	total: numpy 1d array
		total number of inputs per class
	"""
	nz_classes = numpy.nonzero(total)  # nonzero classes
	return numpy.mean(
		correct[nz_classes].astype('float32') / total[nz_classes]
	)

def getLabel(r, g, b):
	if r == 128 and g == 0 and b == 0:
		return 1
	if r == 64 and g == 64 and b == 0:
		return 9
	if r == 128 and g == 128 and b == 0:
		return 5
	if r == 64 and g == 0 and b == 128:
		return 7
	if r == 64 and g == 64 and b == 128:
		return 4
	if r == 128 and g == 64 and b == 128:
		return 2
	if r == 128 and g == 128 and b == 128:
		return 0
	if r == 192 and g == 192 and b == 128:
		return 6
	if r == 0 and g == 0 and b == 192:
		return 3
	if r == 192 and g == 128 and b == 128:
		return 8
	if r == 0 and g == 128 and b == 192:
		return 10
	if r == 0 and g == 0 and b == 0:
		return 11
	return -1

if len(sys.argv) != 3:
	print "Usage: " + sys.argv[0] + " directoryWithCRFOutputs GroundTruthDirectory"
	exit(-1)
	
numberOFClasses = 12

correct = numpy.zeros((numberOFClasses), dtype='int32')
total = numpy.zeros((numberOFClasses), dtype='int32')

care_classes = numpy.ones((numberOFClasses), dtype='int8')
care_classes[11] = 0
gtList = os.listdir(sys.argv[2])
for file in os.listdir(sys.argv[1]):
	print file
	im = Image.open(sys.argv[1] + "/" + file)
	crfOutput = numpy.array(im)
	gtFileName = file[0:-4] + ".png"
	if gtFileName in gtList:
		gt = Image.open(sys.argv[2] + "/" + gtFileName)
		groundTruth = numpy.array(gt)
		
		img_up = numpy.zeros((crfOutput.shape[0], crfOutput.shape[1]))
		curr_y = numpy.zeros((groundTruth.shape[0], groundTruth.shape[1]))
		for i in range(groundTruth.shape[0]):
			for j in range(groundTruth.shape[1]):
				curr_y[i,j] = getLabel(groundTruth[i,j,0],groundTruth[i,j,1],groundTruth[i,j,2])
				img_up[i,j] = getLabel(crfOutput[i,j,0],crfOutput[i,j,1],crfOutput[i,j,2])
		
		for i in range(numberOFClasses):
			if numpy.any(curr_y == i):
				correct[i] += numpy.sum(numpy.equal(img_up[curr_y == i],
                                          curr_y[curr_y == i]))
				total[i] += numpy.sum(curr_y == i)

		
		print groundTruth.shape

care_correct = correct * care_classes
care_total = total * care_classes
class_accuracy = calc_class_accuracy(care_correct, care_total)

total_acc = (numpy.sum(care_correct, dtype='float32') /
			 numpy.sum(care_total)) * 100.

print 'total pixel accuracy {:f} %%'.format(total_acc)
print 'mean class accuracy: {:f} %%'.format(class_accuracy * 100.)
print 'per class accuracies: {}'.format(correct.astype('float32') / total)