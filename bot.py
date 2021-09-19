import pyautogui as gui
from cv2 import cv2
import numpy as np
import os

gui.PAUSE = 0.01

number_images = []
for filename in os.listdir(r".\res\numbers"):
    img = cv2.cvtColor(
        cv2.imread(os.path.join(r".\res\numbers", filename)), cv2.COLOR_BGR2GRAY
    )
    if img is not None:
        number_images.append((img, filename[: filename.index(".")]))


hists = []
for image, name in number_images:
    hists.append((cv2.calcHist([image], [0], None, [256], [0, 256]), name))

box = gui.locateOnScreen(r".\res\anchor.png", confidence=0.95)
if box is None:
    raise BaseException("Please show the main menu")

LEFT_TOP = (box.left, box.top + box.height)
WEIGHT_HIGHT = (2496 - 1327, 1027 - 324)
game_region = (*LEFT_TOP, *WEIGHT_HIGHT)

if 1:
    play_button = gui.locateOnScreen(r".\res\play_button.png", confidence=0.95)
    if play_button is None:
        restart_button = gui.locateOnScreen(
            r".\res\restart_button.png", confidence=0.95
        )
        if restart_button is None:
            raise BaseException("Please show the main menu")
        gui.click(restart_button.left, restart_button.top)
        gui.sleep(3)

    play_button = gui.locateOnScreen(r".\res\play_button.png", confidence=0.95)
    gui.click(play_button.left + 30, play_button.top + 10)
    gui.sleep(1)
    new_game = gui.locateOnScreen(r".\res\new_game_button.png", confidence=0.95)
    if new_game is not None:
        gui.click(new_game.left, new_game.top)
    gui.sleep(1)


cv2.namedWindow('circles')
cv2.moveWindow('circles', 2560 // 2, 20)

maxR = 90
for level_number in range(1, 100):
    shot = gui.screenshot(region=game_region)
    open_cv_image = np.array(shot)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    # cv2.imshow("level", open_cv_image)
    cv2.waitKey(1)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    circles_image = open_cv_image.copy()
    coords = [LEFT_TOP[0] + 150, LEFT_TOP[1] + 200]


    
    while True:
        circles = cv2.HoughCircles(
            gray[90:, 90:], cv2.HOUGH_GRADIENT, 1.65, 40, maxRadius=maxR, minRadius=max(maxR - 40, 10)
        )[0]
        if circles[:, 2].max() / circles[:, 2].mean() > 1.2 or len(circles) * 2 < level_number:
            maxR -= 3
            continue
        break
    # while
    circles[:, 0:2] += 90
    if circles is not None:
        # Get the (x, y, r) as integers
        rounded_circles = np.round(circles[:]).astype("int")
        # loop over the circles
        for (x, y, r) in rounded_circles:
            cv2.circle(circles_image, (x, y), r, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("circles", circles_image)
    cv2.waitKey(1)
    left_side = float("inf")
    right_side = -float("inf")
    top_side = float("inf")
    down_side = -float("inf")

    left_side = min(circles[:, 0])
    right_side = max(circles[:, 0])
    top_side = min(circles[:, 1])
    down_side = max(circles[:, 1])
    # print(circles)
    # print(left_side, right_side, top_side, down_side)

    cv2.rectangle(
        circles_image,
        (int(left_side), int(top_side)),
        (int(right_side), int(down_side)),
        color=(0, 0, 255),
    )
    # cv2.imshow("circles", circles_image)
    circles_mean_size = circles[:, 2].mean()
    field_pos = circles.copy()[:, :2]
    field_pos[:, 0] -= left_side
    field_pos[:, 1] -= top_side
    # print(field_pos)
    field_pos[:, 0] /= right_side - left_side
    field_pos[:, 1] /= down_side - top_side
    # print(field_pos)

    # try to find field size
    cv2.imshow("circles", circles_image)
    cv2.waitKey(1)
    field_size = None
    for i in range(1, 10):
        for j in range(1, i + 1):
            tmp = field_pos.copy()
            tmp[:, 0] *= i
            tmp[:, 1] *= j
            tmp = np.round(tmp)
            un = np.unique(tmp, axis=0)
            if len(tmp) == len(un):
                field_size = (j + 1, i + 1)
                break
        else:
            continue
        break

    if field_size is None:
        raise BaseException("Can't find field size")

    # print(field_size)
    field_pos[:, 0] *= field_size[1] - 1
    field_pos[:, 1] *= field_size[0] - 1

    field_pos = np.round(field_pos).astype("int")
    # print(field_pos)

    field = np.ones(field_size, dtype=np.int8) * -2

    for cell in field_pos:
        field[cell[1], cell[0]] = 1

    # print(field)

    def get_pos_on_screen(row, column):
        return (
            LEFT_TOP[0]
            + left_side
            + column * (right_side - left_side) / (field_size[1] - 1),
            LEFT_TOP[1] + top_side + row * (down_side - top_side) / (field_size[0] - 1),
        )

    def get_pos_on_shot(row, column):
        return (
            left_side + column * (right_side - left_side) / (field_size[1] - 1),
            top_side + row * (down_side - top_side) / (field_size[0] - 1),
        )


    def find_numbers(value, check_this_cell=None):
        for i in range(field_size[0]):
            for j in range(field_size[1]):
                if  ((check_this_cell is not None) and not check_this_cell[i, j]):
                    continue
                cv2.drawMarker(
                    circles_image, tuple(map(int, get_pos_on_shot(i, j))), (100, 100, 100)
                )
                coords = get_pos_on_screen(i, j)
                shot_size = circles_mean_size * 0.7
                shot = gui.screenshot(
                    region=(
                        coords[0] - shot_size,
                        coords[1] - shot_size,
                        2 * shot_size,
                        2 * shot_size,
                    )
                )
                shot_np = np.array(shot)
                # Convert RGB to BGR
                # shot_np = shot_np[:, :, ::-1].copy()
                shot_np = cv2.resize(shot_np, (100, 100))
                shot_np = cv2.cvtColor(shot_np, cv2.COLOR_RGB2GRAY)
                histogram1 = cv2.calcHist([shot_np], [0], None, [256], [0, 256])
                difference_list = set()
                for number_image, number_name in hists:

                    c = cv2.compareHist(histogram1, number_image, cv2.HISTCMP_HELLINGER)
                    # diff = cv2.subtract(shot_np, number_image)
                    # print(number_name, diff.sum())
                    difference_list.add((c, number_name))
                

                minimum = min(difference_list)
                if minimum[1] == '-2':
                    field[i][j] = int(minimum[1])
                elif minimum[1] == '-1':
                    field[i][j] = int(minimum[1])

                elif minimum[0] < value:
                    field[i][j] = int(minimum[1])
            
                else:
                    # cv2.imshow("how_name_it?", shot_np)
                    cv2.waitKey(1)
                    num = input(f"Enter number on picture in {i}:{j} maybe it's {minimum[1]}: ")
                    unic = 0
                    while os.path.isfile(
                        f"res\\numbers\{num}.{i}.{j}.{level_number}({(unic:=unic+1)}).png"
                    ):
                        pass
                    cv2.imwrite(
                        f"res\\numbers\{num}.{i}.{j}.{level_number}({unic}).png", shot_np
                    )
                    number_images.append((shot_np, num))
                    field[i][j] = int(num)

                # cv2.imshow(f"{i}:{j}", shot_np)
                cv2.waitKey(1)

    value = 100
    find_numbers(value)
    cv2.imshow("circles", circles_image)
    cv2.waitKey(1)
    # print(np.where(filed == 0))
    print(field)

    def field_correct(field):
        global value
        un = np.unique(field, return_counts=True)
        only_numbers_count = un[1][2 if un[0][0] == -2 else 1:]
        # print(only_numbers_count > 1)
        # print(np.where(only_numbers_count > 1))
        try:
            wrong_number = (np.where(only_numbers_count > 1)[0])[0]
        except IndexError:
            if len(only_numbers_count) <= max(un[0][2 if un[0][0] == -2 else 1:]):
                wrong_number = max(only_numbers_count)
            else:
                return
        # print(field == (wrong_number))

        while sum(only_numbers_count) > len(only_numbers_count) or len(only_numbers_count) <= max(un[0][2 if un[0][0] == -2 else 1:]):
            find_numbers(value := value * 0.8, field == wrong_number)
            # print(f"{value = }")
            un = np.unique(field, return_counts=True)
            only_numbers_count = un[1][2 if un[0][0] == -2 else 1:]
            # print(only_numbers_count > 1)
            # print(un[0][(2 if un[0][0] == -2 else 1) + (np.where(only_numbers_count > 1)[0])])
            try:
                wrong_number = un[0][(2 if un[0][0] == -2 else 1) + (np.where(only_numbers_count > 1)[0])][0]
            except IndexError:
                if len(only_numbers_count) <= max(un[0][2 if un[0][0] == -2 else 1:]):
                    wrong_number = max(un[0][2 if un[0][0] == -2 else 1:])
                else:
                    return
            # raise BaseException('Incorrecct number identification')



    field_correct(field)
    def find_path(filed):
        current_number = 0
        start = np.where(filed == 0)
        L = [(start[0][0], start[1][0])]
        # print(len(np.where(filed != -2)[0]))
        number_of_cells = len(np.where(filed != -2)[0])

        def get_near_cell(i, j):
            if i > 0:
                yield (i - 1, j)
            if j > 0:
                yield (i, j - 1)
            if i + 1 < filed.shape[0]:
                yield (i + 1, j)
            if j + 1 < filed.shape[1]:
                yield (i, j + 1)

        def rec(number_of_cells, current_number):
            if number_of_cells == 1:
                return True
            for new_i, new_j in get_near_cell(*L[-1]):
                if (
                    (new_i, new_j) in L
                    or filed[new_i, new_j] == -2
                    or filed[new_i, new_j] - current_number > 1
                ):
                    continue
                number_of_cells -= 1
                L.append((new_i, new_j))
                if filed[new_i, new_j] - current_number == 1:
                    current_number += 1
                if rec(number_of_cells, current_number):
                    return True
                L.pop()
                if filed[new_i, new_j] == current_number:
                    current_number -= 1
                number_of_cells += 1
            return False

        rec(number_of_cells, current_number)
        return L


    for i, row in enumerate(field):
        for j, item in enumerate(row):
            if item == -2:
                cv2.putText(circles_image, '#', tuple(map(int, get_pos_on_shot(i, j))), 4, 2, (0, 0, 0), 3)
            elif item == -1:
                cv2.putText(circles_image, '~', tuple(map(int, get_pos_on_shot(i, j))), 4, 2, (0, 0, 0), 3)
            else:
                cv2.putText(circles_image, f'{item}', tuple(map(int, get_pos_on_shot(i, j))), 4, 2, (0, 0, 0), 3)


    cv2.imshow("circles", circles_image)
    cv2.waitKey(1)
    print(field)
    path = find_path(field)
    if len(path) <= 1:
        raise BaseException("can't find path")
    for _ in range(10):
        gui.click(get_pos_on_screen(*path[0]))
        for cell in path:
            coord = get_pos_on_screen(*cell)
            gui.dragTo(*coord, duration=0.02)
        gui.dragTo(*get_pos_on_screen(*path[0]), duration=0.015)
        # gui.click(*get_pos_on_screen(*path[-1]), duration=0.1)
        gui.sleep(0.1)
        end_coord = gui.locateOnScreen(r".\res\end_button.png", confidence=0.95)
        if end_coord is not None:
            gui.click(end_coord.left, end_coord.top, duration=0.05)
            gui.sleep(2)
            send_score = gui.locateOnScreen(r".\res\send_score.png", confidence=0.95)
            gui.click(send_score.left, send_score.top, duration=0.05)
            gui.sleep(0.1)
            gui.write('andrewkraevskii', interval=0.5)
            gui.sleep(0.5)
            write_button = gui.locateOnScreen(r".\res\write.png", confidence=0.95)
            gui.click(send_score.left, send_score.top, duration=0.5)
            gui.sleep(0.05)
            quit()
        button_coord = gui.locateOnScreen(r".\res\next_level_button.png", confidence=0.95)
        if button_coord is None:
            continue
        gui.sleep(0.05)
        gui.click(button_coord.left, button_coord.top, duration=0.05)
        gui.sleep(0.1)
        break

