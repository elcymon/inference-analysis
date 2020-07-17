import pandas as pd
from zipfile import ZipFile
import re
from io import StringIO
import ntpath
import cv2 as cv
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from glob import glob

def getPointAngle(pt,centre,degrees=True):
    angle = np.arctan2(pt[1] - centre[1], pt[0] - centre[0])
    
    if degrees:
        return 180 * angle / np.pi
    
    return angle
# Function to find the circle on 
# which the given three points lie 
def findCircle(x1, y1, x2, y2, x3, y3) :
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = (((sx13) * (x12) + (sy13) *
        (x12) + (sx21) * (x13) +
        (sy21) * (x13)) // (2 *
        ((y31) * (x12) - (y21) * (x13))))
            
    g = (((sx13) * (y12) + (sy13) * (y12) +
        (sx21) * (y13) + (sy21) * (y13)) //
        (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
        2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c 

    # r is the radius 
    r = round(sqrt(sqr_of_r), 5)
    pt1_angle = getPointAngle((x1,y1),(h,k))
    pt2_angle = getPointAngle((x3,y3),(h,k))

    return (round(h),round(k)),round(r),pt1_angle,pt2_angle
def shortenNetworkName(name):
    if '608' in name:
        return 'YOLOv3'
    elif '128' in name:
        return 'tYOLOv3-128'
    elif '224' in name:
        return 'tYOLOv3-224'
    elif '124' in name:
        return 'mSSD-124'
    elif '220' in name:
        return 'mSSD-220'

def drawBoxes(boxesDF,frame):

    for box in boxesDF.index:
        color = (255,0,0)
        x1,y1,x2,y2 = boxesDF.loc[box,['x1','y1','x2','y2']]
        if 'info' in boxesDF.columns:
            if 'FP' in boxesDF.loc[box,'info'] or \
                'FN' in boxesDF.loc[box,'info'] or \
                    'TP' in boxesDF.loc[box,'info']:#none ground truth data
                if 'FP' in boxesDF.loc[box,'info'] and 'new' in boxesDF.loc[box,'info']:
                    color = (0,0,255)
                elif 'FN' in boxesDF.loc[box,'info']:
                    color = (255,0,255)
                elif 'TP' in boxesDF.loc[box,'info']:
                    color = (255,0,0)
                else:
                    continue
            else:
                if 'inter' in boxesDF.loc[box,'info']:
                    color = (0,0,255)
                elif 'new' in boxesDF.loc[box,'info']:
                    color= (255,0,255)
            # insert text information
            frame = cv.putText(frame, boxesDF.loc[box,'id'], (x2,y1), \
                                cv.FONT_HERSHEY_PLAIN, 1, color, 2, cv.LINE_8, False)
        cv.rectangle(img=frame,pt1=(x1,y1),pt2=(x2,y2),
                        color=color,thickness=2)
        
    return frame
def maskFrame(frame,horizon):
    center,radius,start_angle,end_angle = findCircle(*horizon)
    start_angle = 0; end_angle = 360

    axes = (radius,radius)
    angle = 0
    color = (255,255,255)
    thickness = -1
    lineType = cv.LINE_AA
    shift = 0
    ellipse_mask = np.zeros_like(cv.cvtColor(frame,cv.COLOR_BGR2GRAY))

    ellipse_mask = cv.ellipse(ellipse_mask,center,axes,angle,start_angle,end_angle,color,thickness,lineType,shift)
    _,horizon_mask = cv.threshold(ellipse_mask,1,255,cv.THRESH_BINARY)
    newframe = cv.bitwise_and(frame,frame,mask = horizon_mask)
    return newframe
def saveFrame(frame,masked,name):
    while True:
        key = cv.waitKey(1)
        if key == ord('s'):
            cv.imwrite('../thesis_detection_figs/' + name + '.jpg',frame)
            cv.imwrite('../thesis_detection_figs/' + name + '-horizon.jpg',masked)
            break
        elif key == ord('p'):
            break

def raw_detection_video(videoname,networkname,horizon,size=(640,360)):
    video = cv.VideoCapture('../data/mp4/' + videoname + '.MP4')
    frameNo = 1
    zipfile = ZipFile('../data/' + networkname + '.zip')
    ntwkname = shortenNetworkName(networkname)
    while video.isOpened():
        _,frame = video.read()
        if frame is None:
            print('frame is none')
            break
        print(frameNo)
        frame = cv.resize(frame,size)
        if '608' in networkname:
            header = ['class','x1','y1','x2','y2']
        else:
            header = ['class','confidence','x1','y1','x2','y2']
        filename = "{0}/1r1c/{1}-{2:05d}.txt".format(networkname,videoname,frameNo)
        df = pd.read_csv(zipfile.open(filename),sep=' ',names = header)
        #resize df data
        if df.shape[0] > 0:
            df.loc[:,['x1','x2']] = df.loc[:,['x1','x2']] * 640./960.
            df.loc[:,['y1','y2']] = df.loc[:,['y1','y2']] * 360./540.
            df.loc[:,['x1','y1','x2','y2']] = df.loc[:,['x1','y1','x2','y2']].astype(int)
            frame = drawBoxes(df,frame)
        masked = maskFrame(frame,np.round(horizon * 2/3.).astype(np.int))
        cv.imshow(ntpath.basename(videoname) + '-masked',masked)
        cv.imshow(ntpath.basename(videoname),frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('p') or frameNo == 306:
            # cv.waitKey(0)
            saveFrame(frame,masked,'{}-{}-{:05d}'.format(ntwkname, ntpath.basename(videoname),frameNo) )
        frameNo += 1


def detections_from_model_data(videoname,networkname,horizon,size=(640,360)):
    video = cv.VideoCapture('../data/mp4/' + videoname + '.MP4')
    ntwkname = shortenNetworkName(networkname)
    csvfile = '../data/model_data/{}/{}/'.format(videoname,networkname)
    if '608' in networkname:
        csvfile  += '{}-{}-GT-pruned.csv'.format(videoname,networkname)
    else:
        csvfile  += '{}-{}-detection.csv'.format(videoname,networkname)
    df = pd.read_csv(csvfile,header=[0,1],index_col=0,low_memory=False)
    print(df.shape)
    frameNo = 1
    mask = None
    pause = False
    while video.isOpened():
        _,frame = video.read()
        if frame is None:
            print('frame is none')
            break
        frame = cv.resize(frame,size)
        masked = maskFrame(frame,np.round(horizon * 2/3.).astype(np.int))    
        if frameNo in df.index:
            rowseries = df.loc[frameNo,:].dropna()
            litters = rowseries.index.get_level_values(0).unique()
            
        
            if len(litters) > 0:
                rowdf = rowseries.unstack(level=1)
                rowdf.loc[:,['x1','x2','y1','y2']] = rowdf.loc[:,['x1','x2','y1','y2']].mul(2./3.).astype(int)
                frame = drawBoxes(rowdf,frame)
                masked = drawBoxes(rowdf,masked)
            # print(rowdf['info'],rowdf['info'].str.contains('new'))
            if rowdf['info'].str.contains('new').any() and \
                rowdf['info'].str.contains('inter').any() and \
                rowdf['info'].str.contains('iou').any():
                pause = True
            print(frameNo,len(litters))
        
        cv.imshow(ntpath.basename(videoname) + '-masked',masked)
        cv.imshow(ntpath.basename(videoname),frame)
        key = cv.waitKey(1)
        
        if key == ord('q'):
            break
        if key == ord('p') or pause or frameNo == 306:
            # cv.waitKey(0)
            saveFrame(frame,masked,'{}-{}-{:05d}-info'.format(ntwkname, ntpath.basename(videoname),frameNo) )
            pause = False
        frameNo += 1

def bbox_path(resultspath,outputpath,networks,filterby=None):
    if filterby is not None:
        filterbyDF = pd.read_csv(filterby,header=[0,1],index_col=[0,1])
    # cmap = plt.get_cmap('inferno')
    for ntwk in networks:
        print(ntwk)
        fig = plt.figure(figsize=(16,9))
        ax = fig.gca()
        ax.invert_yaxis()
        ax.set_ylim([0,540])
        ax.set_xlim([0,960])
        ax.set_xticks([])
        ax.set_yticks([])
        for csvfile in glob(resultspath + '/*/*/*' + ntwk + '-detection.csv'):
            video = ntpath.basename(ntpath.dirname(ntpath.dirname(csvfile)))
            print(video)
            csvDF = pd.read_csv(csvfile,header=[0,1],index_col=0,low_memory=False)
            if filterby is not None:
                if video in filterbyDF.index.get_level_values(0).unique():
                    litters = filterbyDF.loc[video,:].index
                    csvDF = csvDF.loc[:,litters]
                else:
                    continue
            else:
                litters = []
            for lit in csvDF.columns.get_level_values(0).unique():
                bbox = csvDF[lit].dropna(axis=0,how='all') #drop nan rows
                bbox.loc[:,'cx'] = ((bbox['x1'] + bbox['x2'])/2.0).astype(int)
                bbox.loc[:,'cy'] = ((bbox['y1'] + bbox['y2'])/2.0).astype(int)
                alpha = 0.2
                marker = '.'
                markerfacecolor='none'
                markersize=4
                
                if bbox['info'].str.contains('inter').any():
                    ax.plot(bbox.loc[bbox['info'] == 'inter','cx'],
                            bbox.loc[bbox['info'] == 'inter','cy'],color='r',marker=marker,
                            markersize=markersize,alpha=alpha,linestyle='',zorder=2)
                if bbox['info'].str.contains('iou').any():
                    ax.plot(bbox.loc[bbox['info'] == 'iou','cx'],
                            bbox.loc[bbox['info'] == 'iou','cy'],color='b',marker=marker,
                            markersize=markersize,alpha=alpha,linestyle='',zorder=1)
                if bbox['info'].str.contains('new').any():
                    ax.plot(bbox.loc[bbox['info'] == 'new','cx'],
                            bbox.loc[bbox['info'] == 'new','cy'],color='k',marker='.',
                            markersize=8,alpha=1,linestyle='',zorder=3)
                if bbox['info'].str.contains('TP').any():
                    ax.plot(bbox.loc[bbox['info'] == 'TP','cx'],
                            bbox.loc[bbox['info'] == 'TP','cy'],color='g',marker=marker,
                            markersize=markersize,alpha=alpha,linestyle='',zorder=2)
                if bbox['info'].str.contains('FN').any():
                    ax.plot(bbox.loc[bbox['info'] == 'FN','cx'],
                            bbox.loc[bbox['info'] == 'FN','cy'],color='r',marker=marker,
                            markersize=markersize,alpha=alpha,linestyle='',zorder=1)
                if '608' in ntwk:
                    #plot last seen location
                    ax.plot([bbox['cx'].iloc[-1]],[bbox['cy'].iloc[-1]],
                                color='g',marker='.',
                            markersize=8,alpha=1,linestyle='',zorder=3)
            plt.show()
            
            fig.savefig(outputpath + '/centre_path_' + ntwk + '.png',bbox_inches='tight')
        # break


if __name__ == '__main__':
    first_last_appearance = '../data/simplified_data/yolo_608_horizon'
    detection_data = '../data/model-data'
    video = '20190111GOPR9029-hflip'
    networkname = ['yolov3-litter_10000-th0p0-nms0p0-iSz608',
                    'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz128',
                    'yolov3-tiny-litter_10000-th0p0-nms0p0-iSz224',
                    'mobilenetSSD-10000-th0p5-nms0p0-iSz124',
                    'mobilenetSSD-10000-th0p5-nms0p0-iSz220']
    horizon = np.array([18,162,494,59,937,143])
    # raw_detection_video(video,networkname[4],horizon)
    # detections_from_model_data(video,networkname[2],horizon)
    bbox_path('../data/model_data','../thesis_detection_figs',
        networkname,filterby='../data/simplified_data/yolo_608_horizon/yolo_608_horizon.csv')
