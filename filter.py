import cv2 

''' Constructs a filter, given an image. 
    You can overlay this filter on the eyes, mouth, forehead, or nose. '''
class Filter: 
    def __init__(self, filepath):
        # read in filter image given filepath
        self.filter = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    def eyes_overlay(self, points, frame):
        # resize the filter image w.r.t. points
        f_width = int(points[7][0] - points[9][0])
        f_height = int(points[10][1] - points[8][1])
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)

        # get transparent region (i.e. alpha > 0.7)
        transparent_region = f_resized[:, :, 3] > 0.7 

        # get boundaries (global coord.)
        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[9], f_width, f_height)
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    def mouth_overlay(self, points, frame):
        # TODO: create overlay function for mouth with any image found online!

    def forehead_overlay(self, points, frame):
        # TODO: create overlay for forehead images (i.e. dog ears or flower crowns!)

    def nose_overlay(self, points, frame, vertical_shift=0):
        # resize the filter image w.r.t. points
        f_width = 100
        f_height = 50
        f_resized = cv2.resize(self.filter, (f_width, f_height), interpolation=cv2.INTER_CUBIC)

        # get transparent region (i.e. alpha > 0.7)
        transparent_region = f_resized[:, :, 3] > 0.7 
        
        # get boundaries (global coord.)
        xmin, xmax, ymin, ymax = self.get_x_y_min_max(points[10], f_width, f_height)
        
        # shift the filter to be on center with the nose
        xmin -= f_width // 2
        xmax -= f_width // 2
        ymin += vertical_shift
        ymax += vertical_shift
        frame[ymin:ymax, xmin:xmax, :][transparent_region] = f_resized[:, :, :3][transparent_region]

    ''' 
    Returns the global coordinates box of the filter with respect to a certain point
    ''' 
    def get_x_y_min_max(self, boundary_point, filter_width, filter_height):
        ymin = int(boundary_point[1])
        ymax = ymin + filter_height
        xmin = int(boundary_point[0])
        xmax = xmin + filter_width
        return xmin, xmax, ymin, ymax
        