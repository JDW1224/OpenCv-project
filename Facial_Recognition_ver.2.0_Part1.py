import cv2
import numpy as np
from os import makedirs
from os.path import isdir

# 얼굴 저장 함수
face_dirs = 'faces/'

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 얼굴 검출 함수
def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 카메라에 따라서 수치를 바꿔야 함!
    faces = face_classifier.detectMultiScale(gray, scaleFactor= 1.05, minNeighbors=6, minSize=(100,100))
    # 얼굴이 없으면 패스!
    if faces is():
        return None
    # 얼굴이 있으면 얼굴 부위만 이미지로 만들고
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    # 리턴
    return cropped_face

# 얼굴만 저장하는 함수
def take_pictures():
    # 해당 이름의 폴더가 없다면 생성
    name = input("사용자 명을 입력하시오: ")
    if not isdir(face_dirs+name):
        makedirs(face_dirs+name)

    # 카메라 ON
    # 젯슨 나노를 사용하기 위해서는 아마도 0대신 -1을 넣어야 함!    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        # 카메라로 부터 사진 한장 읽어 오기
        ret, frame = cap.read()
        # 사진에서 얼굴 검출 , 얼굴이 검출되었다면 
        if face_extractor(frame) is not None:
            
            count+=1
            # 200 x 200 사이즈로 줄이거나 늘린다음
            face = cv2.resize(face_extractor(frame),(200,200))
            # 흑백으로 바꿈
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 200x200 흑백 사진을 faces/얼굴 이름/userxx.jpg 로 저장
            file_name_path = face_dirs + name + '/user' +str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            print("얼굴이 검출되지 않았습니다")
            pass
        
        # 얼굴 사진 150장을 다 얻었거나 enter키 누르면 종료
        if cv2.waitKey(1)==13 or count==150:
            break
    # 카메라를 닫고 생성된 윈도우를 모두 닫는다
    cap.release()
    cv2.destroyAllWindows()
    print('완료')

# 메인함수 선언
if __name__ == "__main__":
    # 사진 저장할 이름을 넣어서 함수 호출
    take_pictures()