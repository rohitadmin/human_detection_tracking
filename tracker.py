from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import FPS
from mylib import config, thread
import time
import numpy as np
import argparse, imutils
import dlib, cv2

t0 = time.time()

def human_detect_track():

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	

	
	print("[INFO] Starting the video..")
	vs = cv2.VideoCapture(args["input"])

	
	writer = None

	
	W = None
	H = None

	
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	
	totalFrames = 0
	total_left = 0
	total_right = 0
	x = []
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		
		if args["input"] is not None and frame is None:
			break

		
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		
		status = "Waiting"
		rects = []

		
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					
					trackers.append(tracker)

		
		else:
			# loop over the trackers
			for tracker in trackers:
				
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			
			else:
				
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				
				if not to.counted:
					
					if direction < 0 and centroid[1] < H // 12:
						total_right += 1
						empty.append(total_right)
						to.counted = True

					
					elif direction > 0 and centroid[1] > H // 12:
						total_left += 1
						empty1.append(total_left)
						to.counted = True
						
					x = []
					# compute the sum of total people inside
					x.append(len(empty1)+len(empty))
					#print("Total people inside:", x)


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		("Left Side", total_left),
		("Right Side", total_right)
		]

		info2 = [
		("Total Unique people inside", x),
		]

                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (190, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


		# show the output frame
		cv2.imshow("Human Detection and Tracking", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		
		totalFrames += 1
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] Total spend time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	
	# issue 15
	if config.Thread:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()

t3= time.time()
total_time= t3-t0

human_detect_track()
print("Total execution time:",total_time)
