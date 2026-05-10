import numpy as np

def detect_corners(
    scan_points,
    robot_x,
    robot_y,
    all_observed_corners,
    window=4,
    min_angle_deg=70,
    max_angle_deg=110,
    min_segment_length=0.05,
    max_gap=0.3,
    max_distance=3.0,
    merge_distance=1
):

    corners = []

    points = np.array(scan_points)

    if len(points) < 2 * window + 1:
        return np.array(corners)

    angles = np.arctan2(
        points[:, 1] - robot_y,
        points[:, 0] - robot_x
    )

    sorted_idx = np.argsort(angles)
    points = points[sorted_idx]

    robot_pos = np.array([robot_x, robot_y])

    for i in range(window, len(points) - window):

        curr_pt = points[i]

        ################################################################
        # Фильтр дальности
        ################################################################

        dist_to_robot = np.linalg.norm(curr_pt - robot_pos)

        if dist_to_robot > max_distance:
            continue

        ################################################################
        # Проверка разрывов скана
        ################################################################

        left_gap = np.linalg.norm(points[i] - points[i - 1])
        right_gap = np.linalg.norm(points[i + 1] - points[i])

        if left_gap > max_gap or right_gap > max_gap:
            continue

        ################################################################
        # Вычисляем векторы
        ################################################################

        left_vec = points[i] - points[i - window]
        right_vec = points[i + window] - points[i]

        norm1 = np.linalg.norm(left_vec)
        norm2 = np.linalg.norm(right_vec)

        if (
            norm1 < min_segment_length
            or norm2 < min_segment_length
        ):
            continue

        ################################################################
        # Вычисляем угол
        ################################################################

        cos_theta = np.dot(left_vec, right_vec) / (norm1 * norm2)

        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = np.degrees(np.arccos(cos_theta))

        ################################################################
        # Ищем углы около 90°
        ################################################################

        if min_angle_deg <= angle <= max_angle_deg:

            too_close = False

            ################################################################
            # Проверяем близость к найденным углам текущего скана
            ################################################################

            for c in corners:
                if np.linalg.norm(curr_pt - c) < merge_distance:
                    too_close = True
                    break

            ################################################################
            # Проверяем близость к уже найденным углам
            ################################################################

            for c in all_observed_corners:
                if np.linalg.norm(curr_pt - c) < merge_distance:
                    too_close = True
                    break

            if not too_close:
                corners.append(curr_pt)

    return corners