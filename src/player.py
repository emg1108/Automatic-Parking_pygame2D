"""Player class

Controls player activities and contains car logic

    Attributes:
        name: SALFIC
        date: 03.06.2021
        version: 0.0.1
"""

import pygame

from pygame import Vector2
from loguru import logger
from math import cos, sin, radians, degrees, copysign
import math
from constants import *
import numpy as np

import path_planning
import path_tracking

def catmull_rom(p0, p1, p2, p3, t):
        """4점 Catmull–Rom 보간 (t∈[0,1])"""
        t2 = t * t
        t3 = t2 * t
        return (
            0.5 * (2*p1[0] + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3),
            0.5 * (2*p1[1] + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        )

def smooth_path(path, resolution_per_segment=3): ## osy
    """Catmull–Rom 방식으로 주어진 경로 path를 부드러운 곡선으로 보간"""
    if len(path) < 4:
        return path[:]
    
    smoothed = []
    for i in range(1, len(path) - 2):
        p0, p1, p2, p3 = path[i - 1], path[i], path[i + 1], path[i + 2]
        for j in range(resolution_per_segment):
            t = j / resolution_per_segment
            smoothed.append(catmull_rom(p0, p1, p2, p3, t))
    
    smoothed.append(path[-1])
    return smoothed

def compute_bezier_control_points(p0, p3, dir1, dir2, dist=80): # dist 늘리면 곡선이 완만해짐.
    # dir1, dir2는 단위벡터 (진입 방향, 진출 방향)
    p1 = (p0[0] + dir1[0]*dist, p0[1] + dir1[1]*dist)
    p2 = (p3[0] - dir2[0]*dist, p3[1] - dir2[1]*dist)
    return p1, p2

def direction_from_points(p_prev, p_next):
    dx, dy = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm)

def cubic_bezier(p0, p1, p2, p3, num_points=20):
    path = []
    for t in np.linspace(0, 1, num_points):
        x = (1-t)**3*p0[0] + 3*(1-t)**2*t*p1[0] + 3*(1-t)*t**2*p2[0] + t**3*p3[0]
        y = (1-t)**3*p0[1] + 3*(1-t)**2*t*p1[1] + 3*(1-t)*t**2*p2[1] + t**3*p3[1]
        path.append((x, y))
    return path


class Player(pygame.sprite.Sprite):
    """Handels player activities

    Args:
        pygame (Sprite): Inherits from pygame Sprite class

    """
    def __init__(self, imagePath, x, y):
        """Creates object of player class

        Note:
            The formula for the physics and as a guidens this source was used:
                - https://rmgi.blog/pygame-2d-car-tutorial.html
        
        Args:
            imagePath (str): contains the path to the default car image
            x (int): contains starting x position
            y (int): contains starting y position
        
        Test:
            * the start position must be (x,y)
            * the deafult image must be set
        """
        pygame.sprite.Sprite.__init__(self)
        self.setImage(imagePath)
        self.rect = self.image.get_rect()
        # Initialiazing of important variables
        self.setStart(x,y)
        #Initialize some max values
        self.length = LENGTH
        self.maxSteering = MAX_STEERING_ANGLE
        self.maxAcceleration = MAX_ACCELARATION
        self.maxVelocity = MAX_VELOCITY
        #Variables for smooth stopping
        self.deaccelaration = DEACCELERATION
        self.reverseAcceleartion = REVERSE_ACCELERATION

        self.honkNoise = pygame.mixer.Sound(HONK)

        self.map = None
        self.tracker = None
        self.auto_mode = False

        self.parking_phase = 0      # 0=입구로, 1=정지 지점으로, 2=완료
        self.entry_cell = None      # 그리드 좌표 (row, col)
        self.stop_cell = None       # 그리드 좌표 (row, col)
    
    def setMap(self, map): ## osy
        self.map = map

    def auto_drive(self, timeDif):
        """
        차량 자동 주행 함수 (수학 좌표계 yaw 사용)

        차량의 angle은 화면 기준 -y 방향을 0도로 간주하며,
        이는 수학 좌표계의 +x 방향과 일치하므로:
            car_angle_math = radians(self.angle) + pi/2

        화면 좌표계 : y가 아래로 증가, -y가 0도 (pygame 기준)
        수학 좌표계 : y가 위로 증가, +x가 0도 (atan2 기준)
        차량 좌표계 : 화면 기준 -y가 전방
        """

        # 1. 차량 중심 위치 및 yaw 변환
        center_x = self.rect.centerx # 픽셀
        center_y = self.rect.centery

        car_angle_screen = math.radians(self.angle)    # 차량 좌표계(화면이랑 동일): -y가 0도
        car_angle_math = car_angle_screen + math.pi / 2      # 차량 좌표계->수학 좌표계: +x가 0도

        # 2. 차량 위치 및 출발 셀 계산 (화면 기준 → 그리드 기준)
        grid = self.map.grid # 그리드
        start = ( # 그리드
            int(center_y // BLOCK_SIZE),
            int(center_x // BLOCK_SIZE)
        )
        car_position = (center_x, center_y) # 픽셀
    
        # 3. 주차 목표 셀 설정
        goal_rect = self.map.goal_rect
        entry_cell = (
            (goal_rect.bottom + BLOCK_SIZE * 3) // BLOCK_SIZE,
            goal_rect.centerx // BLOCK_SIZE
        )
        stop_cell = (
            goal_rect.centery // BLOCK_SIZE,
            goal_rect.centerx // BLOCK_SIZE
        )

        # 4. 경로 재생성 조건 확인
        if (not hasattr(self, 'path_to_draw')
            or not self.path_to_draw
            or getattr(self, 'auto_mode_just_activated', False)):

            # (1) front → entry (yaw 제한 O)
            path1 = path_planning.astar(grid, start, entry_cell, car_angle_math, use_angle_limit=True)
            if not path1:
                print("경로 없음 (front→entry)")
                return

            # (2) entry → stop (yaw 제한 X)
            path2 = path_planning.astar(grid, entry_cell, stop_cell, car_angle_math, use_angle_limit=False)
            if not path2:
                print("경로 없음 (entry→stop)")
                return

            # (3) 차량 중심을 시작점에 삽입
            path1 = [(center_y / BLOCK_SIZE, center_x / BLOCK_SIZE)] + path1
            full_path = path1 + path2[1:]

            # 기존 경로 픽셀화
            pixel_path1 = [(col * BLOCK_SIZE + BLOCK_SIZE / 2,
                            row * BLOCK_SIZE + BLOCK_SIZE / 2)
                        for row, col in path1]
            pixel_path2 = [(col * BLOCK_SIZE + BLOCK_SIZE / 2,
                            row * BLOCK_SIZE + BLOCK_SIZE / 2)
                        for row, col in path2]

            # 연결부 정보
            pixel_path1 = pixel_path1[:-2]
            pixel_path2 = pixel_path2[1:]

            p0 = pixel_path1[-1]
            p3 = pixel_path2[0]
            dir1 = direction_from_points(pixel_path1[-2], pixel_path1[-1])
            dir2 = direction_from_points(pixel_path2[0], pixel_path2[1])
            p1, p2 = compute_bezier_control_points(p0, p3, dir1, dir2)

            # 곡선 생성
            bezier_segment = cubic_bezier(p0, p1, p2, p3, num_points=10)

            # 최종 경로
            full_path = pixel_path1[:-1] + bezier_segment + pixel_path2[1:]
            self.path_to_draw = full_path
            self.tracker = path_tracking.PurePursuit(self.path_to_draw)
            self.auto_mode_just_activated = False

        # 5. 조향각 계산 (수학 yaw 사용)
        steer_rad = self.tracker.compute_steering(car_position, car_angle_math)
        steer_deg = math.degrees(steer_rad)
        clamped = max(-self.maxSteering, min(self.maxSteering, steer_deg))
        self.steering = clamped

        # 6. 목표 지점 도달 시 정지
        if self.map.goal_rect.collidepoint(center_x, center_y):
            self.velocity.y = 0
            self.acceleration = 0
            return

        # 7. 차량 이동 및 조향
        self.forward(timeDif)

        # 8. 주행 경로 히스토리 기록
        self.path_history = getattr(self, 'path_history', []) + [(center_x, center_y)]

    def backward(self, timeDif):
        """Accelerates the player backward
        
        Checks if the car is moving forward or backwards and acts appropiat
            
        Args:
            timeDif (float): contains the time since last frame
        
        Test:
            * acceleration must stay between maxAcceleration and -maxAcceleration
            * the acceleration must increace if velocity.y > 0
        """
        if self.velocity.y < 0:
            self.acceleration = self.reverseAcceleartion
        else:
            self.acceleration += 1 * timeDif
            
        self.acceleration = self.checkLimits(self.acceleration, self.maxAcceleration)

    def forward(self, timeDif):
        """Accelerates the player forward
        
        Checks if the car is moving forward or backwards and acts apropiat

        Args:
            timeDif (float): contains the time since last frame
            
        Test:
            * acceleration must stay between maxAcceleration and -maxAcceleration
            * the acceleration must decrease if velocity.y < 0
        """
        if self.velocity.y > 0:
            self.acceleration = -self.reverseAcceleartion
        else:
            self.acceleration -= 1 * timeDif
        
        self.acceleration = self.checkLimits(self.acceleration, self.maxAcceleration)
    
    def emergencyBrake(self):
        """Stops the car immediately
        
        Test:
            * velocity.y must be 0 at the end
            * did the function change other variables
        """
        self.velocity.y = 0
    
    def noAcceleration(self, timeDif):
        """Slowly decreases the speed of the car
        
        After a decleared value is reached the car is fully stopped to prevent slowly creeping of the car

        Args:
            timeDif (float): contains the time since last frame
        
        Test:
            * the car must be fully stopped after reaching the defined limit
            * the acceleration must have the opposite sign as velocity.y if the defined limit is not reached
        """
        if abs(self.velocity.y) > timeDif * self.deaccelaration:
            self.acceleration = -copysign(self.deaccelaration, self.velocity.y)
        else:
            self.velocity.y = 0
            self.acceleration = 0
        
        self.acceleration = self.checkLimits(self.acceleration, self.maxAcceleration)
    
    def right(self, timeDif):
        """Increases the steering of the car
        
        Args:
            timeDif (float): contains the time since last frame
            
        Test:
            * the steering must be increased by STEERING_DIF (declared in constants.py)
            * the steering must be lower or equal maxSteering
        """
        self.steering += STEERING_DIF * timeDif
        self.steering = self.checkLimits(self.steering, self.maxSteering)
    
    def left(self, timeDif):
        """Decreases the steering of the car
        
        Args:
            timeDif (float): contains the time since last frame
            
        Test:
            * the steering must be decreased by STEERING_DIF (declared in constants.py)
            * the steering must be greater or equal then -maxSteering
        """
        self.steering -= STEERING_DIF * timeDif
        self.steering = self.checkLimits(self.steering, self.maxSteering)
        
    def straight(self):
        """Neutralizes the steering
        
        Test:
            * the steering must be 0 at the end of the function
            * other variables should not be changed
        """
        self.steering = 0
    
    def honk(self):
        """Plays a honk noise
        
        Test:
            * the noise has to be played
            * pygame.mixer needs to be initialized
        """
        self.honkNoise.play()
    
    def checkLimits(self, value, maxValue):
        """Checks if a given value exceeds its given maxValue
        
        Is the given maxValue is reached or exceeded, the given value is set to this maxValue

        Args:
            value (float): specific value of a variable
            maxValue (int): maxvalue of this variable

        Returns:
            Float: Checked value
        
        Test:
            * return value can not be greater or less then max or -maxValue
            * if the value did not exceed- or is not equal to maxValue the returned value should be the same as the given one
        """
        if value >= maxValue:
            value = maxValue
        elif value <= -maxValue:
            value = -maxValue
            
        return value

    def calcTurning(self):
        """Calculates the turning with given formula
        
        Based on this calculation the angle of rotation is calculated.
        Therfore the given image can be rotated.

        Returns:
            Float: Contains the velocity while turning
        
        Test:
            * if steering equals to zero no calculation should be made
            * the turningvelocity must be 0 if the steering is neutral
        """
        turningVelocity = 0
        
        if self.steering:
            turningRadius = self.length / sin(radians(self.steering))
            turningVelocity = self.velocity.y / turningRadius
        else:
            turningVelocity = 0
        
        return turningVelocity
    
    def collisionDetection(self, tiles):
        """Checks for collision between the car and obstacles
        
        Based on masks the player class and the tiles class have

        Args:
            tiles (list): contains all tiles that represant an obstacle

        Returns:
            Bool: True if the player coordinates match one of the tiles coordinates, False otherwise
        
        Test:
            * logging information must be writen in log file
            * hit must be True if the player collides with an obstacle
        """
        hit = False
        if pygame.sprite.spritecollide(self, tiles, False, pygame.sprite.collide_mask):
            hit = True
            logger.info(f"Player hit something! >> Level ended")
        return hit
    
    def checkGoalReach(self, map):
        """Checks if the player rect is inside of the goal rect

        Args:
            goal (Tile): contains the information and rect of the goal

        Returns:
            Bool: True if the rect of player is inside of the rect of goal, False otherwise
        
        Test:
            * logging information must be in log file
            * true must be returned if player is in the goal
        """
        grid_y = self.rect.centery // BLOCK_SIZE
        grid_x = self.rect.centerx // BLOCK_SIZE

        try:
            if map.grid[grid_y][grid_x] == 1:  # 목표 영역 값이 1이라고 가정
                logger.info(f"Player reached the goal (grid-based)")
                return True
        except IndexError:
            pass

        return False
    
    def update(self, timeDif, tiles, goal):
        """Updates the parameters of the player
        
        Increases / Decreases the velocity, position and angle.
        Rotates the image based on the changed variables.
        Calls collision functions to check for collisions.

        Args:
            timeDif (float): contains the time since last frame
            tiles (list): contains all tiles that represant an obstacle
            goal (Tile): contains the information and rect of the goal

        Returns:
            hit(Bool): True if a hit occured, False otherwise
            goal(Bool): True if the player is in the goal, False otherwise
        
        Test:
            * if a angle is equal to zero the image can not rotate
            * velocity, position and angle must changed if the user pressed a valid key
        """
        self.velocity += (0, self.acceleration * timeDif)
        self.velocity.y = self.checkLimits(self.velocity.y, self.maxVelocity)
        
        #Calculating turning radius
        if self.auto_mode:
            self.angle += self.steering * timeDif # 각도는 steering(deg/s) * timeDif
            print("self.steering(deg): ", self.steering)
        else:
            turningVelocity = self.calcTurning()
            self.angle += degrees(turningVelocity) * timeDif

        self.position += self.velocity.rotate(-self.angle) * timeDif # -angle은, 수학 좌표계 기준 각도를 Pygame 좌표계 시각에 맞추기 위한 보정임

        self.rotationImage = pygame.transform.rotate(self.image, self.angle)
        self.rect = self.rotationImage.get_rect()
        
        self.rect.centerx = self.position.x * PIXEL_ALIGNMENT 
        self.rect.centery = self.position.y * PIXEL_ALIGNMENT
        # Updating the mask for upcoming collision detection
        self.mask = pygame.mask.from_surface(self.rotationImage)
        
        hit = self.collisionDetection(tiles)
        goal = self.checkGoalReach(self.map)

        logger.info(f"Velocity:{self.velocity}, Acceleration:{self.acceleration}, Position:{self.position}, Angle:{self.angle}")
        return hit, goal
    
    def setImage(self, imagePath):
        """Setter for the image of the player

        Args:
            imagePath (str): Contains the path to the desired image
        
        Test:
            * the imagePath must be valid
            * the image must be (changed) to the image found in imagePath
        """
        self.image = pygame.image.load(imagePath).convert_alpha()
        self.image = pygame.transform.rotate(self.image, PLAYER_INIT_ROT)
        self.image = pygame.transform.scale(self.image, (PLAYER_WIDTH,PLAYER_HIGHT))
        logger.info(f"Car Image set to {imagePath}")
    
    def setStart(self, grid_x, grid_y): ## osy
        self.rect.centerx = grid_x * BLOCK_SIZE + BLOCK_SIZE // 2
        self.rect.centery = grid_y * BLOCK_SIZE + BLOCK_SIZE // 2
        self.position = Vector2(self.rect.centerx / PIXEL_ALIGNMENT, self.rect.centery / PIXEL_ALIGNMENT)
        self.velocity = Vector2(0.0, 0.0)
        self.acceleration = 0.0
        self.steering = 0.0
        self.angle = 0.0