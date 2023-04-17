import pandas as pd
import os
import cv2
from datetime import datetime

class DataCollector:
    def __init__(self, path='DataCollected'):
        self.path = path
        self.imgList = []
        self.steeringList = []
        self.countFolder = 0
        self.count = 0
        self.create_folder()

    def create_folder(self):
        """Create a new folder to store the images."""
        myDirectory = os.path.join(os.getcwd(), self.path)
        while os.path.exists(os.path.join(myDirectory,f'IMG{str(self.countFolder)}')):
            self.countFolder += 1
        self.newPath = myDirectory +"/IMG"+str(self.countFolder)
        os.makedirs(self.newPath)

    def save_data(self, img, steering):
        """Save the image and steering angle in the folder."""
        now = datetime.now()
        timestamp = str(datetime.timestamp(now)).replace('.', '')
        fileName = os.path.join(self.newPath, f'Image_{timestamp}.jpg')
        cv2.imwrite(fileName, img)
        self.imgList.append(fileName)
        self.steeringList.append(steering)

    def save_log(self):
        """Save the log file when the session ends."""
        rawData = {'Image': self.imgList, 'Steering': self.steeringList}
        df = pd.DataFrame(rawData)
        df.to_csv(os.path.join(self.path, f'log_{str(self.countFolder)}.csv'), index=False, header=False)
        print('Log Saved')
        print('Total Images: ',len(self.imgList))

if __name__ == '__main__':
    collector = DataCollector()
    cap = cv2.VideoCapture(1)
    for x in range(10):
        _, img = cap.read()
        collector.save_data(img, 0.5)
        cv2.waitKey(1)
        cv2.imshow("Image", img)
    collector.save_log()
