__author__ = "Rohit Rane"
#https://code-projects.org/dino-game-in-python-with-source-code/
import os
import sys
import pygame
import random
import speech_recognition as sr
import threading
import queue
from pygame import *

import time ############### 추가된 오픈소스 및 라이브러리
import cv2 ################
import numpy as np ########
import mediapipe as mp ####



pygame.init()

font = pygame.font.Font('./font/PressStart2P-Regular.ttf', 12) # PressStart2P 폰트 지정

scr_size = (width,height) = (600,150)
FPS = 60
gravity = 0.6

black = (0,0,0)
white = (255,255,255)
background_col = (235,235,235)

high_score = 0
gamespeed = 4
is_running = False 
thread = None         
results=False
screen = pygame.display.set_mode(scr_size)
clock = pygame.time.Clock()
pygame.display.set_caption("Dino Run ")

jump_sound = pygame.mixer.Sound('sprites/jump.wav')
die_sound = pygame.mixer.Sound('sprites/die.wav')
checkPoint_sound = pygame.mixer.Sound('sprites/checkPoint.wav')

command_queue = queue.Queue()

######################################## 얼굴인식 코드추가 
cam_running = False
cam_thread = None 
leftgesture = None
rightgesture =None

def cam_for_commands():
    global leftgesture, rightgesture, results
    global cam_running
    global gesture #h
    while cam_running:
        try:
            print("얼굴 인식 대기중...")


            file_name = 'my_face.png' 
            last_save_time = time.time()

            mp_drawing = mp.solutions.drawing_utils #h
            mp_hands = mp.solutions.hands#h

            cap = cv2.VideoCapture(0)
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

            ################################h 손인식 추가
            with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, 
                    min_tracking_confidence=0.5) as hands: 
                
                hand_mook = False
                hand_mook_start_time = 0  
                
                hand_phaa = False
                hand_phaa_start_time = 0
            #################################h
                while cam_running:# 캠 온
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
        
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    ############################# 손인식
                    result_track_hand = hands.process(rgb_frame)
                    results = face_detection.process(rgb_frame)


                    leftgesture, rightgesture = None, None  # 왼손과 오른손 제스처 초기화
                    
                    if result_track_hand.multi_hand_landmarks:
                        for hand_idx, hand_landmarks in enumerate(result_track_hand.multi_hand_landmarks):
                        # 왼손인지 오른손인지 판별
                            hand_type = "Right" if result_track_hand.multi_handedness[hand_idx].classification[0].label == "Right" else "Left"
                        
                            # 각 손가락의 위치에 따라 제스처 결정
                            finger_1 = hand_landmarks.landmark[4].y < hand_landmarks.landmark[2].y
                            finger_2 = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
                            finger_3 = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
                            finger_4 = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
                            finger_5 = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

                            # 주먹(0)과 모든 손가락을 펼친 경우(1)를 판단
                            if not any([finger_2, finger_3, finger_4, finger_5]):
                                gesture = 0  # 주먹
                            elif all([finger_1, finger_2, finger_3, finger_4, finger_5]):
                                gesture = 1  # 모든 손가락을 펼침
                            else:
                                gesture = None

                            # 왼손과 오른손에 따라 제스처 값 할당
                            if hand_type == "Left":
                                leftgesture = gesture
                            else:
                                rightgesture = gesture
                        
                            if hand_mook and (2 <= (time.time() - hand_mook_start_time) <= 5):
                                hand_mook = False
                            if hand_phaa and (2 <= (time.time() - hand_phaa_start_time) <= 5):
                                hand_phaa = False
                        

##############################################################손인식 여기까지



                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            ih, iw, _ = frame.shape
                            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                        int(bboxC.width * iw), int(bboxC.height * ih)

                            face_region = frame[y:y + h, x:x + w]

                            face_region = cv2.resize(face_region, (150, 150)) # fixel 크기

                            center = (x + w // 2, y + h // 2)
                            radius = min(w // 2, h // 2)  
                            mask = np.zeros_like(face_region)
                            cv2.circle(mask, (75, 75),75, (255, 255, 255), -1)  # (fixel 크기 절반)

                            antiresult_face = cv2.bitwise_and(face_region, mask)
                            tmp = cv2.cvtColor(antiresult_face, cv2.COLOR_BGR2GRAY)
                            _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)

                            b, g, r = cv2.split(antiresult_face)
                            rgba = [b,g,r, alpha]
                            result_face = cv2.merge(rgba,4) 

                    frame_resized = cv2.resize(frame, (400, 300))
                    cv2.imshow("Your Face Tracking", frame_resized) #=> 화면에 출력할 필요없음

                    if time.time() - last_save_time >= 0.51:############## 저장 시간 간격 변경가능
                        cv2.imwrite(file_name, result_face)
                        
                        #print(f"왼손 :{leftgesture}_____오른손 : {rightgesture} 0:묵 , 1:빠") #########임시코드(삭제예정)
                        #print(f"{rightgesture} 오른손제스처값 => 0:묵 , 1:빠") #########임시코드(삭제예정)
                        #print(f"{file_name} 이미지가 저장되었습니다.") 
                        
                        last_save_time = time.time()
                    
                    if cv2.waitKey(1) & 0xFF == 27: # ESC 키를 누르면 종료
                        break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print("얼굴 인식 오류:", e)

    print("얼굴 인식 종료")

def start_cam_recognition():
    global cam_thread
    global cam_running
    if not cam_running: 
        cam_running= True 
        print("cam_hi") 
        cam_thread = threading.Thread(target=cam_for_commands) #캠인식프로그램 실행 
        cam_thread.daemon = True # main thread 종료시 얼굴인식 쓰레드도 종료
        cam_thread.start()

######################################## 얼굴인식 코드 끝




#----------------음성 인식 코드
def listen_for_commands():
    global gamespeed
    global is_running 
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while is_running:
            try:
                print("음성 인식 대기중...")  
                audio = r.listen(source, timeout=10)
                command = r.recognize_google(audio, language="ko-KR").lower()
                print("인식된 명령:", command)  
                print(gamespeed)
                command_queue.put(command)
            except sr.UnknownValueError:
                print("인식할 수 없는 음성")
            except sr.RequestError as e:
                print("요청 오류:", e)
        
    print("음성 인식 종료")

def start_voice_recognition():
    global thread 
    global is_running 
    if not is_running: 
        is_running= True
        print("Hi") 
        thread = threading.Thread(target=listen_for_commands)
        thread.daemon = True
        thread.start()
#----------------음성 인식 코드


def load_image(
    name,
    sizex=-1,
    sizey=-1,
    colorkey=None,
    ):

    fullname = os.path.join('sprites', name)
    image = pygame.image.load(fullname)
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)

    if sizex != -1 or sizey != -1:
        image = pygame.transform.scale(image, (sizex, sizey))

    return (image, image.get_rect())

def load_sprite_sheet(
        sheetname,
        nx,
        ny,
        scalex = -1,
        scaley = -1,
        colorkey = None,
        ):
    fullname = os.path.join('sprites',sheetname)
    sheet = pygame.image.load(fullname)
    sheet = sheet.convert()

    sheet_rect = sheet.get_rect()

    sprites = []

    sizex = sheet_rect.width/nx
    sizey = sheet_rect.height/ny

    for i in range(0,ny):
        for j in range(0,nx):
            rect = pygame.Rect((j*sizex,i*sizey,sizex,sizey))
            image = pygame.Surface(rect.size)
            image = image.convert()
            image.blit(sheet,(0,0),rect)

            if colorkey is not None:
                if colorkey is -1:
                    colorkey = image.get_at((0,0))
                image.set_colorkey(colorkey,RLEACCEL)

            if scalex != -1 or scaley != -1:
                image = pygame.transform.scale(image,(scalex,scaley))

            sprites.append(image)

    sprite_rect = sprites[0].get_rect()

    return sprites,sprite_rect

def disp_gameOver_msg(retbutton_image,gameover_image):
    retbutton_rect = retbutton_image.get_rect()
    retbutton_rect.centerx = width / 2
    retbutton_rect.top = height*0.52

    gameover_rect = gameover_image.get_rect()
    gameover_rect.centerx = width / 2
    gameover_rect.centery = height*0.35

    screen.blit(retbutton_image, retbutton_rect)
    screen.blit(gameover_image, gameover_rect)

def extractDigits(number):
    if number > -1:
        digits = []
        i = 0
        while(number/10 != 0):
            digits.append(number%10)
            number = int(number/10)

        digits.append(number%10)
        for i in range(len(digits),5):
            digits.append(0)
        digits.reverse()
        return digits

class Dino():
    def __init__(self,sizex=-1,sizey=-1):
        self.images,self.rect = load_sprite_sheet('dino.png',5,1,sizex,sizey,-1)
        self.images1,self.rect1 = load_sprite_sheet('dino_ducking.png',2,1,59,sizey,-1)
        self.my_face_image = None
        self.last_update_time = time.time() #######
        self.update_interval = 0.5            ##### 업데이트 간격조절
        

        self.rect.bottom = int(0.98*height)
        self.rect.left = width/15
        self.image = self.images[0]
        self.index = 0
        self.counter = 0
        self.score = 0
        self.isJumping = False
        self.isDead = False
        self.isDucking = False
        self.isBlinking = False
        self.movement = [0,0]
        self.jumpSpeed = 11.5

        self.stand_pos_width = self.rect.width
        self.duck_pos_width = self.rect1.width
    
    def update_my_face(self): ########################## 얼굴인식데이터 업데이트 내용
        current_time = time.time()
        if current_time - self.last_update_time > self.update_interval:
            if os.path.exists("my_face.png"):  # 파일 존재 확인
                self.my_face_image = pygame.image.load("my_face.png")
                self.my_face_image = pygame.transform.scale(self.my_face_image, (20, 20))
                self.last_update_time = current_time
            else:
                self.my_face_image = pygame.image.load("my_face2.png")

    def draw(self):
        screen.blit(self.image,self.rect)
        self.update_my_face() #######################얼굴인식데이터 업데이트실행
        if self.my_face_image:

            if not self.isDucking:
                # 서있는 상태에서는 my_face 이미지를 Dino 위에 그리기
                screen.blit(self.my_face_image.convert_alpha(), (self.rect.left + 20, self.rect.top + 4))
            else:
                # 앉아 있는 상태에서는 my_face 이미지를 Dino 위에 그리기
                screen.blit(self.my_face_image.convert_alpha(), (self.rect.left + 36, self.rect.bottom - 30))

    def checkbounds(self):
        if self.rect.bottom > int(0.98*height):
            self.rect.bottom = int(0.98*height)
            self.isJumping = False

    def update(self):
        if self.isJumping:
            self.movement[1] = self.movement[1] + gravity

        if self.isJumping:
            self.index = 0
        elif self.isBlinking:
            if self.index == 0:
                if self.counter % 400 == 399:
                    self.index = (self.index + 1)%2
            else:
                if self.counter % 20 == 19:
                    self.index = (self.index + 1)%2

        elif self.isDucking:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2
        else:
            if self.counter % 5 == 0:
                self.index = (self.index + 1)%2 + 2

        if self.isDead:
           self.index = 4

        if not self.isDucking:
            self.image = self.images[self.index]
            self.rect.width = self.stand_pos_width
        else:
            self.image = self.images1[(self.index)%2]
            self.rect.width = self.duck_pos_width

        self.rect = self.rect.move(self.movement)
        self.checkbounds()

        if not self.isDead and self.counter % 7 == 6 and self.isBlinking == False:
            self.score += 1
            if self.score % 100 == 0 and self.score != 0:
                if pygame.mixer.get_init() != None:
                    checkPoint_sound.play()

        self.counter = (self.counter + 1)

    def is_jumping(self):
        return self.isJumping

class Cactus(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('cacti-small.png',3,1,sizex,sizey,-1)
        self.rect.bottom = int(0.98*height)
        self.rect.left = width + self.rect.width
        self.image = self.images[random.randrange(0,3)]
        self.movement = [-1*speed,0]

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)

        if self.rect.right < 0:
            self.kill()

class Ptera(pygame.sprite.Sprite):
    def __init__(self,speed=5,sizex=-1,sizey=-1):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.images,self.rect = load_sprite_sheet('ptera.png',2,1,sizex,sizey,-1)
        self.ptera_height = [height*0.82,height*0.75,height*0.60]
        self.rect.centery = self.ptera_height[random.randrange(0,3)]
        self.rect.left = width + self.rect.width
        self.image = self.images[0]
        self.movement = [-1*speed,0]
        self.index = 0
        self.counter = 0

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        if self.counter % 10 == 0:
            self.index = (self.index+1)%2
        self.image = self.images[self.index]
        self.rect = self.rect.move(self.movement)
        self.counter = (self.counter + 1)
        if self.rect.right < 0:
            self.kill()


class Ground():
    def __init__(self,speed=-5):
        self.image,self.rect = load_image('ground.png',-1,-1,-1)
        self.image1,self.rect1 = load_image('ground.png',-1,-1,-1)
        self.rect.bottom = height
        self.rect1.bottom = height
        self.rect1.left = self.rect.right
        self.speed = speed

    def draw(self):
        screen.blit(self.image,self.rect)
        screen.blit(self.image1,self.rect1)

    def update(self):
        self.rect.left += self.speed
        self.rect1.left += self.speed

        if self.rect.right < 0:
            self.rect.left = self.rect1.right

        if self.rect1.right < 0:
            self.rect1.left = self.rect.right

class Cloud(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self,self.containers)
        self.image,self.rect = load_image('cloud.png',int(90*30/42),30,-1)
        self.speed = 1
        self.rect.left = x
        self.rect.top = y
        self.movement = [-1*self.speed,0]

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self):
        self.rect = self.rect.move(self.movement)
        if self.rect.right < 0:
            self.kill()

class Scoreboard():
    def __init__(self,x=-1,y=-1):
        self.score = 0
        self.tempimages,self.temprect = load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
        self.image = pygame.Surface((55,int(11*6/5)))
        self.rect = self.image.get_rect()
        if x == -1:
            self.rect.left = width*0.89
        else:
            self.rect.left = x
        if y == -1:
            self.rect.top = height*0.1
        else:
            self.rect.top = y

    def draw(self):
        screen.blit(self.image,self.rect)

    def update(self,score):
        score_digits = extractDigits(score)
        self.image.fill(background_col)
        for s in score_digits:
            self.image.blit(self.tempimages[s],self.temprect)
            self.temprect.left += self.temprect.width
        self.temprect.left = 0


def introscreen():

    global is_running, cam_running, results
    temp_dino = Dino(44,47)
    temp_dino.isBlinking = True
    gameStart = False

    temp_ground,temp_ground_rect = load_sprite_sheet('ground.png',15,1,-1,-1,-1)
    temp_ground_rect.left = width/20
    temp_ground_rect.bottom = height

    logo,logo_rect = load_image('logo.png',300,140,-1)
    logo_rect.centerx = width*0.6
    logo_rect.centery = height*0.6

    start_voice_recognition() #음성 인식 시작
    start_cam_recognition() ########################### 캠시작

    voice_recognition_started = False # 음성 인식 시작했다는 것을 표시하기 위한 bool 변수 
    cam_recognition_started = False # 음성 인식 시작했다는 것을 표시하기 위한 bool 변수

    while not gameStart:

        if is_running and not voice_recognition_started:
            voice_recognition_started = True

        if results and not cam_recognition_started:
            cam_recognition_started = True
            
        command = None
        # 음성 명령 처리
        if not command_queue.empty():
            command = command_queue.get()
        if command == "시작" and is_running and results:
            gameStart = True

        if pygame.display.get_surface() == None:
            print("Couldn't load display surface")
            return True
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_UP :
                        temp_dino.isJumping = True
                        temp_dino.isBlinking = False
                        temp_dino.movement[1] = -1*temp_dino.jumpSpeed

        if cam_running:
            if leftgesture == 0 and not temp_dino.is_jumping():  # 왼손 주먹 제스처
                temp_dino.isJumping = True
                temp_dino.movement[1] = -1*temp_dino.jumpSpeed

            if rightgesture == 0:  # 오른손 주먹 제스처
                temp_dino.isDucking = True

            if rightgesture == 1:  # 오른손 펼침 제스처
                temp_dino.isDucking = False

        temp_dino.update()

        if pygame.display.get_surface() != None:
            screen.fill(background_col)
            if voice_recognition_started:
                # 텍스트 렌더링
                voice_text = font.render('VOICE RECOGNITION', True, (255, 0, 0))
                screen.blit(voice_text, (50, 10))

            if cam_recognition_started:
                # 텍스트 렌더링
                cam_text = font.render('CAM RECOGNITION', True, (255, 50, 0))
                screen.blit(cam_text, (320, 10))
            
            screen.blit(temp_ground[0],temp_ground_rect)
            if temp_dino.isBlinking:
                screen.blit(logo,logo_rect)
            temp_dino.draw()

            pygame.display.update()

        clock.tick(FPS)
        if temp_dino.isJumping == False and temp_dino.isBlinking == False:
            gameStart = True

def gameplay():
    global high_score, gamespeed, is_running, thread, leftgesture, rightgesture
    command_queue.queue.clear() # 음성인식이 들어있는 큐 clear
    gamespeed= 4
    startMenu = False
    gameOver = False
    gameQuit = False
    playerDino = Dino(44,47)
    new_ground = Ground(-1*gamespeed)
    scb = Scoreboard()
    highsc = Scoreboard(width*0.78)
    counter = 0
    
    cacti = pygame.sprite.Group()
    pteras = pygame.sprite.Group()
    clouds = pygame.sprite.Group()
    last_obstacle = pygame.sprite.Group()

    Cactus.containers = cacti
    Ptera.containers = pteras
    Cloud.containers = clouds

    retbutton_image,retbutton_rect = load_image('replay_button.png',35,31,-1)
    gameover_image,gameover_rect = load_image('game_over.png',190,11,-1)

    temp_images,temp_rect = load_sprite_sheet('numbers.png',12,1,11,int(11*6/5),-1)
    HI_image = pygame.Surface((22,int(11*6/5)))
    HI_rect = HI_image.get_rect()
    HI_image.fill(background_col)
    HI_image.blit(temp_images[10],temp_rect)
    temp_rect.left += temp_rect.width
    HI_image.blit(temp_images[11],temp_rect)
    HI_rect.top = height*0.1
    HI_rect.left = width*0.73

    while not gameQuit:
        command = None
        # 음성 명령 처리          

        while startMenu:
            pass
        while not gameOver:
            if leftgesture == 0:  # 왼손 제스처가 '주먹'일 때
                if playerDino.rect.bottom == int(0.98*height):
                        playerDino.isJumping = True
                        if pygame.mixer.get_init() != None:
                            jump_sound.play()
                        playerDino.movement[1] = -1*playerDino.jumpSpeed
                    
            if rightgesture == 0:  # 오른손 제스처가 '주먹'일 때
                if not (playerDino.isJumping and playerDino.isDead):
                    playerDino.isDucking = True

            if rightgesture == 1:  # 오른손 제스처가 '보'를 표시할 때
                playerDino.isDucking = False


            if pygame.display.get_surface() == None:
                print("Couldn't load display surface")
                gameQuit = True
                gameOver = True
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gameQuit = True
                        gameOver = True

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE  : #h3space
                            if playerDino.rect.bottom == int(0.98*height):
                                playerDino.isJumping = True
                                if pygame.mixer.get_init() != None:
                                    jump_sound.play()
                                playerDino.movement[1] = -1*playerDino.jumpSpeed

                        if event.key == pygame.K_DOWN: #h2down
                            if not (playerDino.isJumping and playerDino.isDead):
                                playerDino.isDucking = True

                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_DOWN : #h2down
                            playerDino.isDucking = False
            for c in cacti:
                c.movement[0] = -1*gamespeed
                if pygame.sprite.collide_mask(playerDino,c):
                    playerDino.isDead = True
                    if pygame.mixer.get_init() != None:
                        die_sound.play()

            for p in pteras:
                p.movement[0] = -1*gamespeed
                if pygame.sprite.collide_mask(playerDino,p):
                    playerDino.isDead = True
                    if pygame.mixer.get_init() != None:
                        die_sound.play()

            if len(cacti) < 2:
                if len(cacti) == 0:
                    last_obstacle.empty()
                    last_obstacle.add(Cactus(gamespeed,40,40))
                else:
                    for l in last_obstacle:
                        if l.rect.right < width*0.7 and random.randrange(0,50) == 10:
                            last_obstacle.empty()
                            last_obstacle.add(Cactus(gamespeed, 40, 40))

            if len(pteras) == 0 and random.randrange(0,200) == 10 and counter > 500:
                for l in last_obstacle:
                    if l.rect.right < width*0.8:
                        last_obstacle.empty()
                        last_obstacle.add(Ptera(gamespeed, 46, 40))

            if len(clouds) < 5 and random.randrange(0,300) == 10:
                Cloud(width,random.randrange(height/5,height/2))

            playerDino.update()
            cacti.update()
            pteras.update()
            clouds.update()
            new_ground.update()
            scb.update(playerDino.score)
            highsc.update(high_score)

            if pygame.display.get_surface() != None:
                screen.fill(background_col)
                new_ground.draw()
                clouds.draw(screen)
                scb.draw()
                if high_score != 0:
                    highsc.draw()
                    screen.blit(HI_image,HI_rect)
                cacti.draw(screen)
                pteras.draw(screen)
                playerDino.draw()

                pygame.display.update()
            clock.tick(FPS)
            if playerDino.isDead:
                gameOver = True
                if playerDino.score > high_score:
                    high_score = playerDino.score

            if counter%700 == 699:
                new_ground.speed -= 1
                gamespeed += 1

            if not command_queue.empty(): #한 번만 실행되어야 하는데, 계속 인식되어 실행이 되어버림
                command = command_queue.get()

            if command == "빠르게":
                gamespeed += 1  # 게임 속도 증가
                new_ground.speed += 1 #땅의 속도도 맞춰준다.
                break  # 따라서 break를 넣어 명령어를 처리하고 반복문을 빠져나옴
            
            elif command == "느리게":
                gamespeed = max(4, gamespeed - 1)  # 게임 속도 감소, 1 이하로 내려가지 않도록 함
                new_ground.speed =min(-4, new_ground.speed -1) #땅의 속도도 맞춰준다.
                break  # 따라서 break를 넣어 명령어를 처리하고 반복문을 빠져나옴

            counter = (counter + 1)

        if gameQuit:
            is_running=False #음성 인식 종료
            cam_running =False  ############################### 캠 종료
            break

        while gameOver:
            if pygame.display.get_surface() == None:
                print("Couldn't load display surface")
                gameQuit = True
                gameOver = False
            else:

                if not command_queue.empty():
                    command = command_queue.get()
                    if command == "시작":
                        gameOver = False
                        gameplay()         # 게임 오버시 '시작'이라고 음성인식 한다면 게임 재시작

                        
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gameQuit = True
                        gameOver = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            gameQuit = True
                            gameOver = False

                        if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE : #h3space
                            gameOver = False
                            gameplay()
            highsc.update(high_score)
            if pygame.display.get_surface() != None:
                disp_gameOver_msg(retbutton_image,gameover_image)
                if high_score != 0:
                    highsc.draw()
                    screen.blit(HI_image,HI_rect)
                pygame.display.update()
            clock.tick(FPS)

    pygame.quit()
    quit()

def main():
    isGameQuit = introscreen()
    if not isGameQuit:
        gameplay()

main()
