def smoothen(curr_x, curr_y, prev_x, prev_y, factor=10):
    new_x = prev_x + (curr_x - prev_x) / factor
    new_y = prev_y + (curr_y - prev_y) / factor
    return new_x, new_y
