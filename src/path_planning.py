import heapq
import math

# -------------------------------
# 화면 좌표계 vs 수학 좌표계 설명
# -------------------------------
# ▶ 화면 좌표계 (pygame, 이미지 등에서 기본):
#   - (0, 0)은 왼쪽 위
#   - x는 오른쪽으로 증가
#   - y는 아래쪽으로 증가
#
# ▶ 수학 좌표계 (극좌표, atan2 등에서 기본):
#   - (0, 0)은 중심
#   - x는 오른쪽으로 증가
#   - y는 위쪽으로 증가 (화면 기준 y의 반대 방향)
#
# 따라서 화면상의 row 변화량은 y 방향이 반대이므로
# yaw 계산 시 반드시 y 방향을 반전해야 수학 좌표계 기준이 됨

MAX_VIEW_ANGLE_DEG = 30
COS_THRESHOLD = math.cos(math.radians(MAX_VIEW_ANGLE_DEG))
MIN_RADIUS = 1
MAX_RADIUS = 2

def heuristic(a, b):
    """맨해튼 거리 휴리스틱"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_within_view(current, next_pos, yaw, angle_limit=True):
    """
    차량의 진행 방향(yaw) 기준으로 next_pos가 시야 범위 내에 있는지 확인
    yaw은 반드시 수학 좌표계 기준이어야 함 (즉, +x축이 0도, 반시계 방향 양의 각도)
    """
    if not angle_limit:
        return True

    dx = next_pos[1] - current[1]          # col 차이 → x 변화
    dy = next_pos[0] - current[0]      # row 차이 → y (화면 y 증가)

    norm = math.hypot(dx, dy)
    if norm == 0:
        return False

    candidate_vec = (dx / norm, dy / norm)
    heading_vec = (math.cos(yaw), -math.sin(yaw))  # 화면‐수학 변환 시 -sin 유지

    dot = candidate_vec[0] * heading_vec[0] + candidate_vec[1] * heading_vec[1]
    return dot >= COS_THRESHOLD

def simplify_path(path, step=5):
    """경로를 간격 단위로 간단히 줄임"""
    if not path:
        return []
    simplified = [path[0]]
    for i in range(1, len(path)):
        if math.hypot(path[i][0] - simplified[-1][0], path[i][1] - simplified[-1][1]) >= step:
            simplified.append(path[i])
    if path[-1] != simplified[-1]:
        simplified.append(path[-1])
    return simplified

def astar(grid, start, goal, init_yaw, use_angle_limit=True):
    """
    grid: 2차원 리스트 (0 = 장애물, 1 = 통로)
    start, goal: (row, col)
    init_yaw: 라디안, 반드시 수학 좌표계 기준 (+x가 0도, CCW가 양수)
    """
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start], init_yaw))
    visited = set()

    while open_set:
        _, cost, current, path, yaw = heapq.heappop(open_set)

        if current == goal:
            return simplify_path(path, step=4)

        key = (current, round(yaw, 2))
        if key in visited:
            continue
        visited.add(key)

        for dx in range(-MAX_RADIUS, MAX_RADIUS + 1):
            for dy in range(-MAX_RADIUS, MAX_RADIUS + 1):
                if dx == 0 and dy == 0:
                    continue
                dist = math.hypot(dx, dy)
                if not (MIN_RADIUS <= dist <= MAX_RADIUS):
                    continue

                nx = current[0] + dy
                ny = current[1] + dx

                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue
                if grid[nx][ny] == 0:
                    continue

                next_pos = (nx, ny)

                if use_angle_limit and not is_within_view(current, next_pos, yaw):
                    continue

                # 수학 좌표계 기준 yaw 계산: y 방향 반전 필수
                new_yaw = math.atan2(ny - current[1], nx - current[0])

                next_cost = cost + dist
                heapq.heappush(open_set, (
                    next_cost + heuristic(next_pos, goal),
                    next_cost,
                    next_pos,
                    path + [next_pos],
                    new_yaw
                ))

    return None  # 경로 없음
