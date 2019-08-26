from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', action='store_true',
                    help='Use cuda in live demo')
parser.add_argument('--path',type=str, help='Pth of test Dataset *.txt')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX



def predict(net, image_path, transform):
    frame = cv2.imread(image_path)
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if args.cuda :
        x = x.cuda()
    detections = net(x).data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.4:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            det.append((labelmap[i - 1],detections[0, i, j, 0],pt))#cls, score, box
            j += 1         
    return det

if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, BDD100K_CLASSES as labelmap
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
    net = build_refinedet('test', 320, 8)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))
    if args.cuda :
        net = net.cuda()
        cudnn.benchmark = True


    split_lines = open(args.path).read().strip().split()
    for split_line in split_lines :
        img_id = split_line
        f = open('predicted/' + img_id + '.txt' ,'w')
        #image_path = '/home/rvl/Dataset/bdd-data/bdd100k/PASCAL/Validation/JPEGImages/%s.jpg' % img_id#path of load image data
        image_path = '/home/rvl/Downloads/Test_video/JPEGImages/%s.jpg' % img_id#path of load image data
        Detect_list =  predict(net.eval(),image_path, transform) 
        for i in range(0,len(Detect_list)) :
            cls = Detect_list[i][0]
            bnbox = Detect_list[i][2]
            score =  Detect_list[i][1]
            f.write('%s %.2f %d %d %d %d\n'%(cls, score,bnbox[0],bnbox[1],bnbox[2],bnbox[3]))
        f.close()
        print('Complete %s .'%(image_path))
    
