# ESSIL log detection from mov files
# Author: Nick Hoernle
# Date: 09/18/2017
import pdb

import pandas as pd
import numpy as np

import os
import cv2

import copy

#########################################################################
##### HELPER FUNCTIONS
#########################################################################


def get_min_max_in_contour(contour):
    min_x_y = [np.inf, np.inf]
    max_x_y = [0, 0]

    for x,y in np.array(contour)[:,0]:
        if x<min_x_y[0] and y<min_x_y[1]:
            min_x_y = [x,y]

        if x>max_x_y[0] and y<max_x_y[1]:
            max_x_y = [x,y]

    return (min_x_y, max_x_y)

def convert_rec_to_polar(x,y):
    '''
    function to convert rectangular coordinates to polar
    '''
    assert x != 0
    theta = np.tan(y/x)
    r = np.sqrt(x**2 + y**2)
    return r, np.arctan(y/x)

class LogPosition:
    '''
    Program to read in .mov files given as a list and output a Pandas dataframe that gives a r,theta1,theta2 representation of the positions of logs in the connected worlds environment. For a given file, a Log_Position object is created that stores the positions of the logs. The number of logs in the file is assumed to stay constant and the object assumes that when a log is moved, it is moved to the closest position available.

    Important methods that this class provides is get_log_positions, get_log_diff, infer_user_action:

    get_log_positions - returns the raw (r,theta1,theta2) of the logs.
    get_log_diff - returns the diff of for the logs for each timestep allowing one to see where the logs have moved from and to.
    infer_user_action - uses the diff to infer a user action and describes the action in terms of translation (x,y) and rotation (theta) actions.
    '''
    def __init__(self):
        self.mini_video_y_coord = [864, 1080]
        self.mini_video_x_coord = [0, 384]
        self.lower_green = np.array([29,86,6])
        self.upper_green = np.array([64,255,255])
        self.lower_blue = np.array([110,50,50])
        self.upper_blue = np.array([130,255,255])
        self.y_max = 1080 - 864
        self.x_max = 384
        # there is a set number of logs in the system that we can assume is correct.
        self.number_of_logs = 7

    def load_mov_file(self, file_name) :
        '''
        Use OpenCV to read the specified file into a cv2 object.
        '''
        assert os.path.exists(file_name)
        cap = cv2.VideoCapture(file_name)
        self.mov_file = cap
        return self.mov_file

    def __read_next_frame(self):
        assert self.mov_file != None
        returned, frame = self.mov_file.read()
        return returned, frame

    def __extract_boundary(self,original,hsv_image, lower, upper, flag):
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

    def __get_rect_log_representation(self, r, theta1, theta2, length=np.arange(-5,5,0.5)):
        (x,y) = line_centroid(r, theta1)
        points = np.array([[l * np.cos(theta2) + x, l * np.sin(theta2) + y] for l in length])
        return points

    def __detect_logs(self, frame):

        ###############################################
        # Currently using cv2 to detect the mean log position and angle
        # this can and should be improved
        ###############################################
        min_video_frame = frame[self.mini_video_y_coord[0]:self.mini_video_y_coord[1],self.mini_video_x_coord[0]:self.mini_video_x_coord[1]]
        hsv = cv2.cvtColor(min_video_frame, cv2.COLOR_BGR2HSV)
        th3 = self.__extract_boundary(min_video_frame,hsv,self.lower_blue, self.upper_blue,0)

        image ,contours, heirarchy = cv2.findContours(th3,1,2)

        colored = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

        count =0
        black = np.zeros(colored.shape)
        centers=[]

        # split contours
        split_contours = []

        for contour in contours:

            min_x_y, max_x_y = get_min_max_in_contour(contour)

            split_flag = False
            while np.sqrt((min_x_y[0]-max_x_y[0])**2 + (min_x_y[1]-max_x_y[1])**2) > 25: # validate this number
                split_flag = True
                new_contour = []
                old_contour = []
                for point in contour:
                    point = point[0]

                    if point[0]<min_x_y[0]+25 and point[1]<min_x_y[1]+25:
                        new_contour.append([point])
                    else:
                        old_contour.append([point])

                if len(new_contour) > 2:
                    split_contours.append(new_contour)

                contour = old_contour
                if len(contour) == 0:
                    break
                min_x_y, max_x_y = get_min_max_in_contour(contour)

            if (not split_flag) or (len(contour) > 2):
                split_contours.append(contour)

        for contour in split_contours:
            coord_points = np.array([[p[0][0],p[0][1]] for p in contour])

            mu = np.mean(coord_points, axis=0)
            cov_var = np.cov(coord_points.T)

            angle = np.arctan(np.linalg.eig(cov_var)[0][1]/np.linalg.eig(cov_var)[0][0])

            r, theta = convert_rec_to_polar(mu[0],self.y_max-mu[1])
            centers.append(
            {
                'r': r,
                'theta1': theta,
                'theta2': angle,
                'cov_var': cov_var,
                'num_data': len(coord_points)
            })

        self.image = image
        self.centers = centers
        return image,centers

    def __detect_log_positions(self):

        log_centers = []
        mu, cov = None, None

        while self.mov_file.isOpened():
            ret,frame = self.__read_next_frame()

            if ret==True:
                logs,centers = self.__detect_logs(frame)
                log_centers.append(centers)
            else:
                # The next frame is not ready, so we try to read it again
                self.mov_file.set(1, count-1)
                print("something not working")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(100)

            if cv2.waitKey(10) == 27:
                break
            if self.mov_file.get(1) == self.mov_file.get(7):
                break

        self.current_log_centers = log_centers
        return log_centers

    def __match_logs_and_update(self, known_position, updated_sample):

        prior_position = {
            'r': np.sqrt(self.y_max**2 + self.x_max**2),
            'theta1': np.pi/4,
            'theta2': 0,
            'cov_var': np.array([[self.x_max, 0],[0, self.y_max]]),
            'num_data': 0
        }

        # make a prior
        known_position = [copy.deepcopy(prior_position) if pos == None else pos for pos in known_position]

        sample = []
        if len(updated_sample) > self.number_of_logs:

            order = np.argsort([samp['r'] for samp in updated_sample])[::-1]
            updated_sample = [updated_sample[order[i]] for i in range(self.number_of_logs)]

        elif len(updated_sample) < self.number_of_logs:

            for i in range(self.number_of_logs):

                if i < len(updated_sample):
                    sample.append(updated_sample[i])
                else:
                    sample.append(copy.deepcopy(known_position[i]))

            updated_sample = sample

        ordered_sample = []
        for log in known_position:

            distances = [log['r']**2 + samp['r']**2 - 2 * log['r'] * samp['r'] * np.cos(log['theta1'] - samp['theta1']) for samp in updated_sample]
            closest = np.argmin(distances)
            ordered_sample.append(updated_sample[closest])
            del updated_sample[closest]

        # TODO Baysean update to account for noise in the system

        return ordered_sample

    def infer_log_positions_over_time(self):

        log_centers = self.__detect_log_positions()

        fixed_log_centers = [None] * len(log_centers)
        # re_order the coordinates to provide a mapping of one specific log
        for i, log_positions_sample in enumerate(log_centers):

            # we have self.number_of_logs (7) new positions that we are trying to evalate
            if i == 0:
                fixed_log_centers[i] = self.__match_logs_and_update([None]*self.number_of_logs, log_positions_sample)
            else:
                fixed_log_centers[i] = self.__match_logs_and_update(fixed_log_centers[i-1], log_positions_sample)

        self.log_positions = fixed_log_centers
        return self.log_positions

    def get_log_positions_DF(self):
        assert self.log_positions != None

        positions = []

        for i, log in enumerate(self.log_positions):

            position = []
            for pos in log:

                position.append(pos['r'])
                position.append(pos['theta1'])
                position.append(pos['theta2'])
            position.append(i)
            positions.append(position)

        labels = sum([['r_%i'%i, 'theta1_%i'%i, 'theta2_%i'%i] for i in range(7)], []) + ['time']

        log_df = pd.DataFrame.from_records(positions, columns=labels).set_index('time')
        self.log_df = log_df
        return self.log_df

    def get_actions(self):

        if hasattr(self, 'log_df'):
            self.get_log_positions_DF()

        df = self.get_log_positions_DF()

        min_movement_len = 20
        min_rotation_ang = np.pi/4
        min_spin_angle = np.pi/4

        translation = (df[df.columns[df.columns.str.contains('r_')]].diff().abs() > min_movement_len).any(axis=1) | (df[df.columns[df.columns.str.contains('theta1_')]].diff().abs() > min_rotation_ang).any(axis=1)
        rotation = (df[df.columns[df.columns.str.contains('theta2_')]].diff().abs() > min_spin_angle).any(axis=1)
        action = translation | rotation
        return pd.DataFrame({
                    'translation':translation.values,
                    'rotation':rotation.values,
                    'action':action.values,
                    'time':action.index,
                }).set_index('time')

if __name__ == '__main__':

    data_path = '/Volumes/Seagate Backup Plus Drive/connected_worlds_water/'
    interesting_file_path = 'water_level_3/2017-09-19T10_01_14-0.mov'

    log_detector = LogPosition()
    log_detector.load_mov_file(data_path + interesting_file_path)
    log_detector.infer_log_positions_over_time()
    log_detector.get_log_positions_DF()
