from re import L
import numpy as np 

MIN_Y_OVERLAP = 0.5
MAX_X_DIST = 50
OFFSET_RATIO_H = 0.5


def get_height_of_box(box: list = []):
    """ Get height of box

    Args:
        box (list): box format = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    Returns:
        int: height of box
    """
    # print(box)
    return box[3][1] - box[0][1]

def is_on_same_line(box_a, box_b, min_y_overlap_ratio=MIN_Y_OVERLAP):
    """ Check if two boxes are on the same line

    Args:
        box_a (list): box format = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        box_b (list): box format = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        min_y_overlap_ratio (float): min y overlap ratio

    Returns:
        bool: True if two boxes are on the same line
    """

    a_y_min = np.min(np.array(box_a)[:, 1])
    a_y_max = np.max(np.array(box_a)[:, 1])
    b_y_min = np.min(np.array(box_b)[:, 1])
    b_y_max = np.max(np.array(box_b)[:, 1])

    # make sure that box a is alway on top of box b
    if a_y_min > b_y_min: 
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max 

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or overlap >= min_b_overlap

        else:
            return True 
    else:
        False 

def format_to_lines(output_ocr: list = []):
    """_summary_

    Args:
        output_ocr (list): list of item with item = (poly box, text, conf)

    Returns:
        list: list of lines
    """

    merge_lines = []
    x_sorted_boxes = sorted(output_ocr, key=lambda item: np.min(np.array(item[0])[:, 0]))
    skip_ids = set()

    for i in range(len(x_sorted_boxes)):
        if i in skip_ids:
            continue
        rightmost_box_idx = i 
        line = [rightmost_box_idx]
        for j in range(i+1, len(x_sorted_boxes)):
            if j in skip_ids:
                continue 
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx][0], x_sorted_boxes[j][0], min_y_overlap_ratio=MIN_Y_OVERLAP):
                line.append(j)
                skip_ids.add(j)
                rightmost_box_idx = j 

        # split line into lines if the distance between two neightboring sub-lines is greater than max_x_dist
        lines = []
        line_idx = 0 
        lines.append([x_sorted_boxes[line[0]]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k-1]]
            dist = np.min(np.array(curr_box[0])[:, 0]) - np.max(np.array(prev_box[0])[:, 0])
            if dist > MAX_X_DIST:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(x_sorted_boxes[line[k]])
        merge_lines.extend(lines)
    return merge_lines

    

def get_name_from_output(output_ocr: list = []):
    """ Get medicine name from output

    Args:
        output_ocr (list): list of item with item = (poly box, text, conf)

    Returns:
        str: medicine name
    """

    # output_ocr = sorted(output_ocr, key=lambda item: get_height_of_box(item[0]), reverse=True)
    # output_ocr_lines = format_to_lines(output_ocr)
    # output_ocr = [item for line in output_ocr_lines for item in line]

    heights = [get_height_of_box(item[0]) for item in output_ocr]
    max_height = max(heights)
    max_height_index = heights.index(max_height)
    name = output_ocr[max_height_index][1]

    output_names = [output_ocr[max_height_index]]
    for idx in range(max_height_index-1, 0, -1):
        box = output_ocr[idx][0]
        h = get_height_of_box(box)
        #if output_ocr[idx][1] == 'Tears':
        print(h, max_height)
        if h >= max_height * OFFSET_RATIO_H:
            output_names.append(output_ocr[idx])
            # name = output_ocr[idx][1] + " " + name
        else:
            break 

    for idx in range(max_height_index + 1, len(output_ocr)):
        box = output_ocr[idx][0]
        h = get_height_of_box(box)
        print(h, max_height)
        if h >= max_height * OFFSET_RATIO_H:
            # name = name + " " + output_ocr[idx][1]
            output_names.append(output_ocr[idx])
        else:
            break

    name = get_full_text_from_output(output_names)

    return name 
    


    

def get_full_text_from_output(output_ocr: list = [], sep=' '):
    """ Get full text from output

    Args:
        output_ocr (list): list of item with item = (poly box, text, conf)
    
    """
    output_ocr_lines = format_to_lines(output_ocr)
    
    texts = [ " ".join([item[1] for item in line]) for line in output_ocr_lines]
    text = sep.join(texts)
    return text 


    