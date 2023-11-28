import cv2

def main():
    floatti = [15.8, 15.2]
    inte = [0]
    for i in range(2):
        floatti[i] = int(floatti[i])
        print(floatti[i])
    # cap = cv2.VideoCapture(0)
    
    # while True:
    #     suc, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()    