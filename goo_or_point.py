from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse

thresh1 = 10
thresh2 = 10

detection_graph, sess = detector_utils.load_inference_graph()
model_anomaly = 
model_partial = 
ms1 = 
ms2 = 
lof1 = 
lof2 = 
flag = False
status = "none"
matrix = []

def anomaly_detection(img, ms, lof):
    score = model_anomaly.predict(img, batch_size=1)
    score = score.reshape((1,-1))
    score = ms.fit_transform(score)
    return -lof._decision_function(score)    

def anomaly_detection_and_draw(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    global flag, status, matrix
    if (scores[0] > score_thresh):
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                      boxes[i][0] * im_height, boxes[i][2] * im_height)
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        
        # pointer or not
        if status == "none":
            score = anomaly_detection(image_np[int(top):int(bottom), int(left):int(right)], ms1, lof1)
            if score < thresh1:
                flag = True
                status = "pointer"
                matrix.append([int(left), int(top)])
            else:
                status = "none"
                
        # magic or not
        if flag == True and not status == "pointer":
            score = anomaly_detection(image_np[int(top):int(bottom), int(left):int(right)], ms2, lof2)
            if score < thresh2:
                flag = False
                status = "magic"
            else:
                status = "none"
                
        # draw
        if status == "pointer":
            cv2.rectangle(image_np, p1, p2, (77, 77, 255), 3, 1)
        elif status == "magic":
            cv2.rectangle(image_np, p1, p2, (255, 241, 144), 3, 1)
        else: #normal
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            
    # pointer draw
    if flag == True: 
        if len(matrix) > 2:
            xy = np.array(matrix)
            p1 = (int(np.min(xy[:,0])), int(np.min(xy[:,1])))
            p2 = (int(np.max(xy[:,0])), int(np.max(xy[:,1])))
            cv2.rectangle(image_np, p1, p2, (255, 77, 77), 3, 1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.6,#0.2
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    
    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np)
        
        detector_utils.draw_point(cnum_hands_detect, args.score_thresh,
                                         scores, boxes, im_width, im_height,
                                         image_np, matrix)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
