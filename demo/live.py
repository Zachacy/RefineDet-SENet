from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import torch.backends.cudnn as cudnn
from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--video',type=str, help='Test Video')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda in live demo')
parser.add_argument('--record', action='store_true',
                    help='record demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

if args.record == True :
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30.0, (1280,640))



def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if args.cuda :
            x = x.cuda()
        detections = net(x).data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                info = '%s / %.2f' %(labelmap[i - 1], detections[0, i, j, 0])
                cv2.putText(frame, info, (int(pt[0]), int(pt[1])),
                            FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame
        

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = cv2.VideoCapture(args.video)
    # start fps timer
    # loop over frames from the video file stream
    while (stream.isOpened()):
        # grab next frame
        _,frame = stream.read()
        time_update = time.time()
        # update FPS counter
        frame = predict(frame)

        s = "fps : " + str(1/(time.time() - time_update))
        cv2.putText(frame, s, (1280 - 250, 50), 0, 1, (255, 0, 0), 2)

        # keybindings for display
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

        if args.record == True :
            res=cv2.resize(frame,(1280,640),interpolation=cv2.INTER_CUBIC) 
            out.write(res)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # exit
            break
    if args.record == True :
        out.release()


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, KITTI_CLASSES as labelmap
    #from ssd import build_ssd
    from models.refinedet import build_refinedet

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                  CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    #net = build_ssd('test', 300, 21)    # initialize SSD
    net = build_refinedet('test', 320, 4)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    if args.cuda :
        net = net.cuda()
        cudnn.benchmark = True

    
    cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information
    #fps.stop()

    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup

    stream.stop()
    cv2.destroyAllWindows()
