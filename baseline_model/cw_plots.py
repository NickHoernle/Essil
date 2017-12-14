import numpy as np
import scipy as sp
import scipy.stats as stats
import pandas as pd
import copy
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pickle

#just identify water flow path for drawing graphs
def extract_boundary(original,hsv_image, lower, upper, flag):
    # need end points of the boundary too
    mask = cv2.inRange(hsv_image, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(original,original,mask= mask)
    #boundaries in gray scale
    gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding and gaussian filtering  to make the logs white and the background black for better detection
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #logs will be white in th3
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if(flag==1):
        black, extLeft, extRight, cx,cy = find_contour(th3,original)
        return black,extLeft,extRight,cx,cy
    return th3

def detect_water(min_video_frame):
    hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
    # define range of green/yellow color in HSV
    lower_green = np.array([29,86,6])
    upper_green = np.array([64,255,255])
    th3 = extract_boundary(min_video_frame,hsv,lower_green, upper_green,0)
    store = th3
    # morphing to get the skeletal structure/ medial line of the water flow
    size = np.size(th3)
    skel = np.zeros(th3.shape,np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while(not done):
        eroded = cv2.erode(th3,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(th3,temp)
        skel = cv2.bitwise_or(skel,temp)
        th3 = eroded.copy()

        zeros = size - cv2.countNonZero(th3)
        if zeros==size:
            done = True
    return store,skel

def detect_logs(min_video_frame, colour_transition=1000000):
    hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    th3 = extract_boundary(min_video_frame,hsv,lower_blue, upper_blue,0)

    #smooth the logs (current version very fat lines)
    image ,contours, heirarchy = cv2.findContours(th3,1,2)#cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     print(contours)

    #Draw log contour + bonding rects
    colored = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    count =0
    black = np.zeros(colored.shape)
    centers=[]
    for contour in contours:
        coord_points = np.array([[p[0][0],p[0][1]] for p in contour])

        if len(coord_points) < 10:
            continue

        # TODO: if contour is really long we need to split it up

        mu = np.mean(coord_points, axis=0)
        cov_var = np.cov(coord_points.T)

        angle = np.arctan(np.linalg.eig(cov_var)[0][1]/np.linalg.eig(cov_var)[0][0])

#         r, theta = convert_rec_to_polar(mu[0],y_max-mu[1])
#         image = cv2.circle(black,(cx,cy),2,(0,255,0),4)
#         centers.append([r,theta,angle])

    return image,centers

def construct_transformed_image(ts, v_name, fwd_bkwd='bkwd', lam_=0.80, num=15, x_min=0, x_max=384, y_min=864, y_max=1080):

    cap = cv2.VideoCapture(v_name)

    sh_1 = x_max-x_min
    sh_0 = y_max-y_min

    transformed_logs = np.zeros((sh_0, sh_1, 4), dtype=np.float16)
    transformed_water = np.zeros((sh_0, sh_1, 4), dtype=np.float16)

    start1 = 0.2
    start2 = 0.2
    end = 1
    lam1 = (end-start1)/(num)
    lam2 = (end-start2)/(num)
#     print(lam1)
    for t in range(num):

        ts = ts + 1 if fwd_bkwd=='bkwd' else ts - 1

        cap.set(1,ts+6) # I took 6 units from the csv for a cleaner sample
        ret, frame = cap.read()

        water, skel = detect_water(frame[y_min:y_max, x_min:x_max])
        logs, centers = detect_logs(frame[y_min:y_max, x_min:x_max])

        #######################################################################
        ### View the logs and the water
        #######################################################################
        start1 += 0.01
        start2 += 0.01
#         lam1 += 0.01
#         lam2 += 0.01

        # TODO: depending on fwd_bkwd, change the alpha param
        for i, row in enumerate(logs):
            for j, col in enumerate(row):

                if logs[i,j]:
                    for l in range(4):
                        if transformed_logs[i,j,l] == 0:
                            transformed_logs[i,j,l] = start1;

                    transformed_logs[i,j,0] += lam1
                    transformed_logs[i,j,1] = 0
                    transformed_logs[i,j,2] = 0
                    transformed_logs[i,j,3] += lam1

                if water[i,j]:
                    for l in range(4):
                        if transformed_water[i,j,l] == 0:
                            transformed_water[i,j,l] = start2;
                    transformed_water[i,j,0] = 0
                    transformed_water[i,j,1] = 0
                    transformed_water[i,j,2] = 255
                    transformed_water[i,j,3] += lam2

        #######################################################################
        ### View the logs and the water
        #######################################################################

    return transformed_logs, transformed_water

def get_relevant_e_vec(phi, waters=[np.array([1,0,0,0,0])]):

    for i in range(120):
        next_ = phi.dot(waters[-1])
        waters.append(next_/np.sum(next_))

    return waters[-1]

def plot_log_positions(df_name, results, names, data_csvs, data_csvs_raw, path, x_min=0, x_max=384, y_min=864, y_max=1080):
    # for each split in found splits
    video_name = names[int(df_name.split('_')[1])-1]
    print('File:', video_name)
    video_name = video_name.replace('.csv', '.mov')

    col_names = ['Other_Water', 'Desert_Water', 'Jungle_Water', 'Wetlands_Water', 'Plains_Water', 'Reservoir_Water']

    e_vecs = []
    for i, res in enumerate(results[df_name]['result_params']):
        waters = [np.zeros(len(res['Phi']))]
        waters[0][0] = 1
        e_vecs.append(get_relevant_e_vec(res['Phi'], waters=waters))

    for i, time_split in enumerate(results[df_name]['breaks']):

        if time_split not in [0, len(data_csvs[df_name].values)]:

            min_ = time_split // 60
            sec_ = time_split - (min_ * 60)

            print("Split @ %i:%02d min" % (min_, sec_))

            print('***********POSSIBLE THINGS TO NOTICE******************')

            for k in range(1, 5):

                if e_vecs[i][k] - e_vecs[i-1][k] > 0.05:
                    print('Increase in %s water'%(col_names[k]))
                elif e_vecs[i][k] - e_vecs[i-1][k] < -0.05:
                    print('Decrease in %s water'%(col_names[k]))
            print()
            print('******************************************************')

            fig, axes = plt.subplots(1,2, figsize=(15,20))
            ax = axes[0]

            num_back = 20
            ts = time_split - num_back
            min1 = ts // 60
            sec1 = ts - (min1 * 60)

            min2 = time_split // 60
            sec2 = time_split - (min2 * 60)

            ax.set_title("%i:%02d-%i:%02d min (before split)" % (min1, sec1, min2, sec2))
            lam_ = 0.75
            thresh = 0.45

            (t_logs, t_water) = construct_transformed_image(ts,
                                        lam_=lam_,
                                        v_name = path+video_name,
                                        fwd_bkwd='bkwd',
                                        num=num_back )
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            t_water[t_water > 1] = 1
            t_logs[t_logs > 1] = 1
            t_water[t_water[:,:,3] < thresh] = 0
            ax.imshow(t_water)
            ax.imshow(t_logs)

            y_corr = 15
            x_corr = -135

            trees = []
            for biome in ['Desert', 'Plains', 'Jungle', 'Wetlands']:
                trees.append(data_csvs_raw[df_name].iloc[ts][['%s_lv1'%biome, '%s_lv2'%biome, '%s_lv3'%biome, '%s_lv4'%biome]].sum())

            ann00 = ax.annotate("Desert Trees: %i" % (trees[0]), xy=[0,0], xytext=[x_min-x_min,y_min+y_corr-y_min], xycoords='data')
            ann01 = ax.annotate("Plains Trees: %i" % (trees[1]), xy=[0,0], xytext=[x_max+x_corr-x_min,y_min+y_corr-y_min], xycoords='data' )
            ann02 = ax.annotate("Jungle Trees: %i" % (trees[2]), xy=[0,0], xytext=[x_max+x_corr-x_min,y_max-y_min], xycoords='data')
            ann03 = ax.annotate("Wetland Trees: %i" % (trees[3]), xy=[0,0], xytext=[x_min-x_min,y_max-y_min], xycoords='data')

            ax = axes[1]

            ts = time_split+num_back
            min3 = ts // 60
            sec3 = ts - (min3 * 60)

            ax.set_title("%i:%02d-%i:%02d min (after split)" % (min2, sec2, min3, sec3))

            (t_logs, t_water) = construct_transformed_image(ts,
                                        v_name = path+video_name,
                                        fwd_bkwd='fwd',
                                        num=num_back )
            ax.set_aspect('equal')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            t_water[t_water > 1] = 1
            t_logs[t_logs > 1] = 1

            t_water[t_water[:,:,3] < thresh] = 0
            ax.imshow(t_water)
            ax.imshow(t_logs)

            trees = []
            for biome in ['Desert', 'Plains', 'Jungle', 'Wetlands']:
                trees.append(data_csvs_raw[df_name].iloc[ts][['%s_lv1'%biome, '%s_lv2'%biome, '%s_lv3'%biome, '%s_lv4'%biome]].sum())


            ann10 = ax.annotate("Desert Trees: %i" % (trees[0]), xy=[0,0], xytext=[x_min-x_min,y_min+y_corr-y_min], xycoords='data')
            ann11 = ax.annotate("Plains Trees: %i" % (trees[1]), xy=[0,0], xytext=[x_max+x_corr-x_min,y_min+y_corr-y_min], xycoords='data')
            ann12 = ax.annotate("Jungle Trees: %i" % (trees[2]), xy=[0,0], xytext=[x_max+x_corr-x_min,y_max-y_min], xycoords='data')
            ann13 = ax.annotate("Wetland Trees: %i" % (trees[3]), xy=[0,0], xytext=[x_min-x_min,y_max-y_min], xycoords='data')

            plt.show()

            print('--------------------------------------------------------')
            print('--------------------------------------------------------')
            print()
