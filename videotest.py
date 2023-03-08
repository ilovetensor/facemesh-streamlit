import cv2

video = cv2.VideoCapture('demo.mp4')
   
if (video.isOpened() == False): 
    print("Error reading video file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
    
while video.isOpened():
    ret, frame = video.read()
  
    if ret == True: 
  
        # Write the frame into the
        # file 'filename.avi'
        result.write(frame)
  
        # Display the frame
        # saved in the file
        cv2.imshow('Frame', frame)
  
        # Press S on keyboard 
        # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
  
    # Break the loop
    else:
        break
  
# When everything done, release 
# the video capture and video 
# write objects
video.release()
result.release()

    

# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")
