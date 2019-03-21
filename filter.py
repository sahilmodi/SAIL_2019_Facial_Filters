import cv2 

''' Constructs a filter, given an image. 
    You can overlay this filter on the eyes, mouth, forehead, or nose. '''
class Filter: 
    def __init__(self, filepath):
        self.filter = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)


    def eyes_overlay(self, points, frame):
        f_width = int(points[7][0] - points[9][0])
        f_height = int(points[10][1] - points[8][1])
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)
        transparent_region = f_resized[:, :, 3] > 0.7 

        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[9], f_width, f_height)
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    def mouth_overlay(self, points, frame):
        f_width = int(points[11][0] - points[12][0])
        f_height = int(points[14][1] - points[13][1])
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)
        transparent_region = f_resized[:, :, 3] >0.7 

        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[12], f_width, f_height)
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    def forehead_overlay(self, points, frame):
        f_width = int(points[7][0] - points[9][0])
        f_height = int(points[10][1] - points[8][1])
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)
        transparent_region = f_resized[:, :, 3] > 0.7 

        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[9], f_width, f_height)
        ymin -= 100
        ymax -= 100
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    def nose_overlay(self, points, frame, vertical_shift=0):
        f_width = 100
        f_height = 50
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)
        transparent_region = f_resized[:, :, 3] > 0.7 
        
        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[10], f_width, f_height)
        xmin -= f_width // 2
        xmax -= f_width // 2
        ymin += vertical_shift
        ymax += vertical_shift
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    def get_x_y_min_max(self, boundary_point, f_width, f_height):
        ymin = int(boundary_point[1])
        ymax = ymin + f_height
        xmin = int(boundary_point[0])
        xmax = xmin + f_width
        return xmin, xmax, ymin, ymax
        