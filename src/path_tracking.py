import math

class PurePursuit:
    """
    성능 개선 Pure Pursuit
    – yaw : 수학 좌표계 기준 (+x=0°, CCW 양수)
    – path: [(x,y)] 픽셀 좌표 리스트
    – lookahead: 고정 lookahead 스텝 수
    – gain: 조향 반응 강도
    """

    def __init__(self, path, lookahead=3, gain=10):
        self.path      = path
        self.lookahead = lookahead
        self.gain      = gain
        self._last_idx = 0
        self._target_idx = 0

    def compute_steering(self, position, car_angle_math):
        # 1) 최근점(closest) 인덱스 탐색 (뒤로 튀지 않도록 캐싱)
        idx = self._last_idx
        min_d = math.hypot(position[0]-self.path[idx][0],
                           position[1]-self.path[idx][1])
        for i in range(self._last_idx, len(self.path)):
            d = math.hypot(position[0]-self.path[i][0],
                           position[1]-self.path[i][1])
            if d < min_d:
                min_d = d
                idx = i
        self._last_idx = idx

        # 2) lookahead 인덱스 계산
        self._target_idx = min(idx + self.lookahead, len(self.path)-1)
        tx, ty = self.path[self._target_idx]

        # 2. 목표점과 차량 위치 차이
        dx = tx - position[0]
        dy = ty - position[1]

        # y축 방향 보정 (화면 → 수학 좌표계)
        dy_math = -dy

        # 3. 목표점 방향, 차량 방향
        path_angle = math.atan2(dy_math, dx)
        angle_error = path_angle - car_angle_math
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # 4. 조향각(비례제어)
        steer = self.gain * angle_error
        return steer