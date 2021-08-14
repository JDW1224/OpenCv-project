# OpenCv-project
2021 cbnu contest

출처 : https://www.cnblogs.com/AdaminXie/p/9010298.html

파일의 용량이 큰 관계로 data/data_dlib 에 https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat 을 다운로드 받아서 넣어야 함 


필요환경:

dlib

numpy

scikit-image

pandas


1. get_faces_camera.py 실행 

	n을 누르면 새로운 사람을 저장할 수 있는 파일이 생성되고 s를 누르면 얼굴 사진이 저장됨

2. features_extraction_to_csv 실행

	1과정에서 저장된 얼굴 사진에서 dlib를 활용하여 얼굴 특징 파일을 뽑아냄(csv)

3. face_reco_from_camera_ot_multi_people 실행

	카메라를 사용하여 얼굴 인식을 함

	
