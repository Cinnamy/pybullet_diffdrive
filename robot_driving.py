motorIdx = [0, 1, 2, 3]

SPEED = 8.0
d = 2.0

#DISTANCE_FRONTSIDE = 2.83
DISTANCE_FRONTSIDE = 1.4
DISTANCE_TURN = 2.5
DISTANCE_STOP = 1.5
DISTANCE_SIDE = 0.9

def calculate_wheel_speeds_square(
    dist_0,
    dist_45,
    dist_90,
    dist_270,
    dist_315,
    y
):

    # Поворачиваем в углах направо
    if dist_0 <= DISTANCE_TURN:
        return [d, SPEED, d, SPEED]

    # Не врезаемся в левую стену
    if (
        dist_270 < DISTANCE_SIDE
    ):
        return [d, SPEED, d, SPEED]

    # Не уходим далеко от левой стены
    if (
        dist_270 > DISTANCE_SIDE + 0.1
    ):
        return [SPEED, d, SPEED, d]

    # Иначе едем вперёд
    return [SPEED, SPEED, SPEED, SPEED]


def calculate_wheel_speeds_snake(
    dist_0,
    dist_45,
    dist_90,
    dist_270,
    dist_315,
    y
):

    if (
        dist_0 < DISTANCE_STOP
    ):
        print('0')
        if (dist_90 < dist_270):
            return [SPEED, d, SPEED, d]
        else:
            return [d, SPEED, d, SPEED]

    # Поворачиваем змейкой
    if (
        dist_45 > DISTANCE_FRONTSIDE
        and y >= DISTANCE_TURN    
    ):
        print('1')
        return [d, SPEED, d, SPEED]

    if (
        dist_315 > DISTANCE_FRONTSIDE
        and y <= -DISTANCE_TURN
    ):
        print('2')
        return [SPEED, d, SPEED, d]

    # Не врезаемся в левую стену
    if (
        dist_90 > DISTANCE_SIDE
        #and dist_270 < DISTANCE_SIDE
    ):
        print('3')
        return [d, SPEED, d, SPEED]

    # Не врезаемся в правую стену
    if (
        dist_90 < DISTANCE_SIDE
    ):
        print('4')
        return [SPEED, d, SPEED, d]

    # Иначе едем вперёд
    print('5')
    return [SPEED, SPEED, SPEED, SPEED]