# Demo RegisterDeviceNotification etc.  Creates a hidden window to receive
# notifications.  See serviceEvents.py for an example of a service doing
# that.

from __future__ import unicode_literals
import sys, time
import win32gui, win32con, win32api, win32file
import win32gui_struct, winnt
import subprocess
from skimage.filters import threshold_adaptive
import numpy as np
import cv2
import imutils
import os
import pytesseract
from matplotlib import pyplot as plt
import pyttsx
import glob
from PIL import Image
from nltk import tokenize
import serial
from textblob import TextBlob
import language_check



'''detection service starts here'''

# These device GUIDs are from Ioevent.h in the Windows SDK.  Ideally they
# could be collected somewhere for pywin32...
GUID_DEVINTERFACE_USB_DEVICE = "{A5DCBF10-6530-11D2-901F-00C04FB951ED}"

myDevice = str("DEV_BROADCAST_INFO:{'devicetype': 5, 'name': '\\\\\\\?\\\\USB#VID_03F0&PID_5607#0404110000006475#{a5dcbf10-6530-11d2-901f-00c04fb951ed}', 'classguid': IID('{A5DCBF10-6530-11D2-901F-00C04FB951ED}')}")

abcd = 1
# WM_DEVICECHANGE message handler.
def OnDeviceChange(hwnd, msg, wp, lp):
    # Unpack the 'lp' into the appropriate DEV_BROADCAST_* structure,
    # using the self-identifying data inside the DEV_BROADCAST_HDR.
    info = win32gui_struct.UnpackDEV_BROADCAST(lp)
    #print "Device change notification:", wp, str(info)
    #print str(info)
    #print myDevice
    if wp != 7:
        
        if wp == 32768:
            print ("Device inserted")
            if str(info) == myDevice:
                print ("Our Device detected")
                abcd = subprocess.Popen("notepad.exe")
            else:
                print ("its not our device")
        else:
            global abcd
            abcd.terminate()
            print ("Device removed")

    if wp==win32con.DBT_DEVICEQUERYREMOVE and info.devicetype==win32con.DBT_DEVTYP_HANDLE:
        # Our handle is stored away in the structure - just close it
        print ("Device being removed - closing handle")
        win32file.CloseHandle(info.handle)
        # and cancel our notifications - if it gets plugged back in we get
        # the same notification and try and close the same handle...
        win32gui.UnregisterDeviceNotification(info.hdevnotify)
    return True


def TestDeviceNotifications(dir_names):
    wc = win32gui.WNDCLASS()
    wc.lpszClassName = 'test_devicenotify'
    wc.style =  win32con.CS_GLOBALCLASS|win32con.CS_VREDRAW | win32con.CS_HREDRAW
    wc.hbrBackground = win32con.COLOR_WINDOW+1
    wc.lpfnWndProc={win32con.WM_DEVICECHANGE:OnDeviceChange}
    class_atom=win32gui.RegisterClass(wc)
    hwnd = win32gui.CreateWindow(wc.lpszClassName,
        'Testing some devices',
        # no need for it to be visible.
        win32con.WS_CAPTION,
        100,100,900,900, 0, 0, 0, None)

    hdevs = []
    # Watch for all USB device notifications
    filter = win32gui_struct.PackDEV_BROADCAST_DEVICEINTERFACE(
                                        GUID_DEVINTERFACE_USB_DEVICE)
    hdev = win32gui.RegisterDeviceNotification(hwnd, filter,
                                               win32con.DEVICE_NOTIFY_WINDOW_HANDLE)
    hdevs.append(hdev)
    # and create handles for all specified directories
    for d in dir_names:
        hdir = win32file.CreateFile(d, 
                                    winnt.FILE_LIST_DIRECTORY, 
                                    winnt.FILE_SHARE_READ | winnt.FILE_SHARE_WRITE | winnt.FILE_SHARE_DELETE,
                                    None, # security attributes
                                    win32con.OPEN_EXISTING,
                                    win32con.FILE_FLAG_BACKUP_SEMANTICS | # required privileges: SE_BACKUP_NAME and SE_RESTORE_NAME.
                                    win32con.FILE_FLAG_OVERLAPPED,
                                    None)

        filter = win32gui_struct.PackDEV_BROADCAST_HANDLE(hdir)
        hdev = win32gui.RegisterDeviceNotification(hwnd, filter,
                                          win32con.DEVICE_NOTIFY_WINDOW_HANDLE)
        hdevs.append(hdev)

    # now start a message pump and wait for messages to be delivered.
    print ("Watching", len(hdevs), "handles - press Ctrl+C to terminate, or")
    print ("add and remove some USB devices...")
    if not dir_names:
        print ("(Note you can also pass paths to watch on the command-line - eg,")
        print ("pass the root of an inserted USB stick to see events specific to")
        print ("that volume)")
    while 1:
        win32gui.PumpWaitingMessages()
        time.sleep(0.01)
    win32gui.DestroyWindow(hwnd)
    win32gui.UnregisterClass(wc.lpszClassName, None)






'''Maincode starts here'''

pushlist =[]
pointer =0
pointer1 =0
poplist =[]
ser = serial.Serial('/dev/ttyUSB0', 9600)


def poplist_function(text):
    global poplist
    global pointer1
    global pushlist
    global pointer
   
    poplist.append(text)








    pointer1 =pointer1 +1

def emptyfunction():
    global poplist
    global pointer1
    global pushlist
    global pointer
    global ser
    if (pointer1>=0):
        pointer1 =pointer1 -1
        subprocess.call('echo '+poplist[pointer1]+'|festival --tts', shell=True)
        pointer = pointer +1
        print ("this is pointer in empty ") 
        print (pointer)
        x= ser.inWaiting()
        print (x)
        if (x != 0):
            print ("went into loop")
            serVal = ser.readline()
            print (serVal)
            if('D' in serVal):
                print ("pause is pressed")
                serVal = ser.readline()
                if('C' in serVal):
                    pop_function()
                else :
                    emptyfunction()
                
            else :
                pop_function()


  


def pop_function():
    global poplist
    global pointer1
    global pushlist
    global pointer
    pointer = pointer-1
    print ("this is pointer in pop") 
    print (pointer)
    text = pushlist[pointer]
    poplist_function(text)
    emptyfunction()

def push_function(text):
    global poplist
    global pointer1
    global pushlist
    global pointer
    pushlist.append(text)
    pointer = pointer +1
    print ("this is pointer")
    print (pointer)



def Camera_Capture(N=1):
    cap = cv2.VideoCapture(1)
    ret = cap.set(3, 1920)
    ret = cap.set(4,1080)
    ret,frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()
    return frame

def init():
    os.system('v4l2-ctl -d /dev/video1 -c brightness=180')
    os.system('v4l2-ctl -d /dev/video1 -c contrast=100')
    os.system('v4l2-ctl -d /dev/video1 -c saturation=0')
    os.system('v4l2-ctl -d /dev/video1 -c sharpness=220')

def Capture_Vlc_Img():
    a='vlc -I dummy v4l2:///dev/video1:width=1920:height=1080 --video-filter scene --no-audio --scene-path /home/gopi --scene-prefix image_prefix --scene-format tiff --scene-replace  vlc://quit --run-time=6'
    os.system(a)

def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape
    edges = cv2.Canny(image, 150, 200, 3, 5)
    # ret3,edges = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=width / 2.0, maxLineGap=20)
    show(lines[10])
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,minLineLength=width / 2.0)
    angle = 0.0
    nlines = lines.size
    for x1, y1, x2, y2 in lines[0]:
        angle += np.arctan2(y2 - y1,x2 - x1)
    # print angle*180
    return angle*180

def deskew(img): 
    image = cv2.bitwise_not(img)
    image= cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_CONSTANT)
    angle=compute_skew(img)
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    # theta=np.cos(angle*np.pi/180)
    maxHeight,maxWidth =  image.shape[:2]
    M = cv2.getRotationMatrix2D(center,angle, 1.0)
    rotated = cv2.warpAffine(image, M, (maxWidth,maxHeight), flags=cv2.INTER_CUBIC)
    lite=cv2.getRectSubPix(rotated, (maxWidth,maxHeight), center)
    image = cv2.bitwise_not(lite)
    return image

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    # warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight),flags=cv2.INTER_CUBIC)
    return warped

def total_col(warped_new,col,y1_row,y2_row):
    total=[]
    for i in range(0,col):
        j=0
        for a in warped_new[y1_row:y2_row,i]:
            if a < 255:
                j=j+1
        total.append(j)
    return total

def total_row(warped_new,row,x1_col,x2_col):
    total=[]
    for i in range(0,row):
        j=0
        for a in warped_new[i,x1_col:x2_col]:
            if a < 255:
                j=j+1
        total.append(j)
    return total

def Row_Cropping_init(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    threshold = col/N
    i=0
    while i <= row-1:
        if total[i]>threshold:
            while total[i]>threshold and i<row-1:
                # print "breaking_pt1 loop, i =" + str(i)+ "total is " + str(total[i]) + " threshold is " + str(threshold)
                if i==row-1:
                    break
                i+=1
            breaking_pt1_row.append(i)
            while total[i]<threshold and i<row-1:
                # print "breaking_pt2 loop, i =" + str(i)+ "total is " + str(total[i]) + " threshold is " + str(threshold)
                i+=1
                if i==row-1:
                    break
            breaking_pt2_row.append(i)
        i+=1
    return breaking_pt1_row,breaking_pt2_row

def Column_Cropping_init(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    threshold=row/N
    # threshold_2 = 0
    i=0
    while i<col-1:
        if total[i]>threshold :
            while total[i]>threshold and i<col-1:
                # print "breaking_pt1 loop, i =" + str(i)+ "total is " + str(total[i]) + " threshold is " + str(threshold)
                i+=1
                if i==col-1:
                    break
            breaking_pt1_row.append(i)
            if i==col-1:
                break
            while total[i]<threshold and i<col-1:
                # print "breaking_pt2 loop, i =" + str(i)+ "total is " + str(total[i]) + " threshold is " + str(threshold)
                i+=1
                if i==col-1:
                    break
            breaking_pt2_row.append(i)
        i+=1
    return breaking_pt1_row,breaking_pt2_row

def show(image,Name='Outline'):
    cv2.imshow(Name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Row_Cropping(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    row_new=col/N
    i=0
    while i<row-1:
        i+=1
        # print " outer col"
        # print i
        if total[i]<row_new:
            while total[i]<row_new and i<row-1:
                # print "breaking_pt1 loop, i =" + str(i)
                i+=1
                if i==row-1:
                    break
            breaking_pt1_row.append(i)
            while total[i]>row_new and i<row-1:
                # print "breaking_pt2 loop, i =" + str(i)
                i+=1
                if i==row-1:
                    break
            breaking_pt2_row.append(i)
    return breaking_pt1_row,breaking_pt2_row

def Column_Cropping(total,row,col,N=10):
    breaking_pt1_row=[]
    breaking_pt2_row=[]
    row_new=row/N
    i=0
    while i<col:
        # print " outer col"
        # print i
        # print "total = " + str(total[i])  
        if total[i]<row_new:
            while total[i]<row_new and i<col:
                # print "breaking_pt1 loop, i = " + str(i)+ "total = "+str(total[i])+"threshold = "+str(row_new)
                i+=1
                if i==col:
                    break
            breaking_pt1_row.append(i)
            while i<col and total[i]>row_new:
                # print "breaking_pt2 loop, i =" + str(i)+ "total = "+str(total[i])+" threshold = "+str(row_new)
                i+=1
                if i==col:
                    break
            breaking_pt2_row.append(i)
        i+=1
    return breaking_pt1_row,breaking_pt2_row

def speak(image):
    a=pytesseract.image_to_string(Image.open(image))
    #print a
    #print ("ikkada nunchi tesuko ra rei vinnava ledha")
    #print unidecode(a)
    #tyu =re.sub(r'[^\x00-\x7F]+',' ',a)
    #print tyu
    #a=a.encode('ascii',errors='ignore')
    #a =''.join([i if ord(i) < 128 else ' ' for i in a])
    sentences_list = tokenize.sent_tokenize(a.decode("ascii", 'ignore'))
    #print sentences_list
    
    for text in  sentences_list :
        cleaned_text = text.replace('\n','')
        #cleaned_text = ["\" + x for x in cleaned_text.split()]
        cleaned_text ='"' + cleaned_text +'"'
        print (cleaned_text)
        text_blob=TextBlob(cleaned_text)
        text_blob_correct=text_blob.correct()
        text_str=str(text_blob_correct)
        #print text_str
        tool=language_check.LanguageTool('en-US')
        matches = tool.check(text_str)
        text_f=language_check.correct(text_str, matches)
        push_function(text_f)
        subprocess.call('echo '+text_f+'|festival --tts', shell=True)
        if (ser.inWaiting() != 0):
            serVal = ser.readline()
            if('D' in serVal):
                print ("pause is pressed")
                serVal = ser.readline()
                if('C' in serVal):
                    pop_function()
                elif('D' in serVal):
                    continue
            elif('C' in serVal):
                pop_function()

def kernel(N=3,Type=1):
    kernel = np.zeros((N,N), dtype=np.uint8)
    if Type==1 :
        ''' Vertical kernel '''
        kernel[:,(N-1)/2] = 1
        return kernel
    if Type==2 :
        ''' Horizontal Kernel '''
        kernel[(N-1)/2,:] = 1
        return kernel
    if Type==3 :
        ''' Star Kernel  '''
        kernel[:,(N-1)/2] = 1
        kernel[(N-1)/2,:] = 1
        return kernel
    if Type==4 :
        ''' Box Kernel '''
        kernel = np.ones((N,N),np.uint8)
        return kernel

def Serial_Cap():
    a=glob.glob('/dev/ttyUSB*')    
    ser = serial.Serial(a[0], 9600)
    while (1):
      serVal = ser.readline()
      print (serVal)
      if ('A' in serVal):
          Capture_Vlc_Img()
          # image = Camera_Capture()
          break
    # return image

def Bounding(image,X=20):
    row,col,lol=image.shape
    image[0:X,:,:]=[0,0,0]
    image[row-X:row,:,:]=[0,0,0]
    image[:,0:X,:]=[0,0,0]
    image[:,col-X:col,:]=[0,0,0]
    return image

def PreProcessing(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5), 0)
    ret3,th3 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def Bounding_Box(th3,x=0.02):
    (_,cnts, _) = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
    cnts=sorted(cnts,key=cv2.contourArea, reverse=True)[:5]
    c=cnts[0]
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,x*peri,True)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if len(approx)==4 :
        box=approx
    return box

def Scanned(orig,box,threshold=251,offset_adaptive=10,ratio=1):
    warped =four_point_transform(orig,box.reshape(4,2)*ratio)
    warped = cv2.bilateralFilter(warped,9,75,75) #Bilateral Filtering
    warped =cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ret2,th2 = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # warped = threshold_adaptive(warped, threshold, offset = offset_adaptive)
    # warped = warped.astype("uint8")*255
    return th2,warped

def Crop(warped_new,warped):
    row,col=warped_new.shape
    # print " row = "+str(row)+"col = " + str(col)    
    total = total_col(warped_new,col,0,row)
    breaking_pt1_col,breaking_pt2_col=Column_Cropping(total,row,col,10)
    # print " breaking_pt1 = " + str(breaking_pt1_col)
    # print " breaking_pt2 = " + str(breaking_pt2_col)
    k=0
    for i in range(len(breaking_pt1_col)):
        diff = breaking_pt2_col[i]-breaking_pt1_col[i]
        if diff>col/20:
            lol=warped[:,breaking_pt1_col[i]:breaking_pt2_col[i]]
            cv2.imwrite("crop_"+str(k)+".jpg",lol)
            # if abs(compute_skew(lol)<10):
            #     cv2.imwrite("crop_"+str(k)+".jpg",lol)
            # else :
            #     lol=deskew(lol)
            #     cv2.imwrite("crop_"+str(k)+".jpg",lol)
            k=k+1
    return k

def Sharpen(image):
    kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                 [-1,2,2,2,-1],
                                 [-1,2,8,2,-1],
                                 [-1,2,2,2,-1],
                                 [-1,-1,-1,-1,-1]])/8.0
    output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
    return output_3


'''Capture Serial Input and Image '''

init()

image = Serial_Cap()
# Capture_Vlc_Img()
# image = Camera_Capture()
''' Read Image '''

image =  cv2.imread('image_prefix.tiff')
# os.system('rm image_prefix.tiff')
# image =  cv2.imread('1.png')
orig = image
show(image)
''' Create Initial Boundary  '''

image = Bounding(image)

''' PreProcessing '''

image =  PreProcessing(image)
show(image)
''' Bounding Box Coorinates '''

box=Bounding_Box(image)
print ('lol')
''' Scanned Output '''

th2,warped = Scanned(orig,box,51,10)
show(th2,'Otsu')
show(warped,'Scanned')
'''Erosion operations on otsu image '''
warped_new=th2
warped_new = cv2.dilate(warped_new,kernel(N=3,Type=4),iterations=1) 
warped_new = cv2.erode(warped_new,kernel(N=5,Type=4),iterations=10)
# show(warped_new,'Erosion')
# warped_new = cv2.erode(warped,kernel(N=5),iterations=10)

''' Crop '''

k = Crop(warped_new,warped)
print (k)
''' Speak '''

for i in range(0,k):
    speak("crop_"+str(i)+".jpg")






if __name__=='__main__':
    # optionally pass device/directory names to watch for notifications.
    # Eg, plug in a USB device - assume it connects as E: - then execute:
    # % win32gui_devicenotify.py E:
    # Then remove and insert the device.
    TestDeviceNotifications(sys.argv[1:])
