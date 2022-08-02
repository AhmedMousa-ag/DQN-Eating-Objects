
class player():
    def __init__(self,co_object):
        self.co = co_object
        #self.current_coordin = ()
        #self.initate_pos(acti_num=start_pos_action)
        pass
    def initate_pos(self,acti_num):
        # 0 = North , 1 = East , 2 = West , 3 = South , 4 = Middle
        north_pos = (0,int(self.co.col / 2))
        east_pos = (int(self.co.row / 2), self.co.col -1)
        west_pos = (int(self.co.row/2),0)
        south_pos = (self.co.row-1,int(self.co.col/2))
        middl_pos = (int(self.co.row / 2), int(self.co.col / 2))
        #-------------------------
        if acti_num == 0:
            self.current_coordin = north_pos
        elif acti_num == 1:
            self.current_coordin = east_pos
        elif acti_num == 2:
            self.current_coordin = west_pos
        elif acti_num == 3:
            self.current_coordin = south_pos
        else:
            self.current_coordin = middl_pos
        return self.current_coordin
    # ---------------------------------
    def get_curr_pos(self):
        return self.current_coordin
    # ---------------------------------
    def action_taker(self,action):
        # 0 = Up , 1 = Right , 2 = Left , 3 = Down , 4 = Do Nothing
        if action == 0:
            self.current_coordin = self.move_up()
        elif action == 1:
            self.current_coordin = self.move_right()
        elif action == 2:
            self.current_coordin = self.move_left()
        elif action == 3:
            self.current_coordin = self.move_down()
        else:
            pass

        return self.current_coordin
    # ---------------------------------
    def move_up(self):
        row,col = self.current_coordin
        row = row-1
        if self.co.is_coordinates_within_vertical_range((row, col)):
            self.co.remove_player_prev_pos(self.current_coordin)
            self.co.add_player_to_map((row,col))
            self.current_coordin = (row, col)
        return self.current_coordin
    # ----------------------------
    def move_right(self):
        row, col = self.current_coordin
        col = col + 1
        if self.co.is_coordinates_within_horizantel_range((row, col)):
            self.co.remove_player_prev_pos(self.current_coordin)
            self.co.add_player_to_map((row,col))
            self.current_coordin = (row, col)
        return self.current_coordin
    # ----------------------------
    def move_left(self):
        row, col = self.current_coordin
        col = col - 1
        if self.co.is_coordinates_within_horizantel_range((row, col)):
            self.co.remove_player_prev_pos(self.current_coordin)
            self.co.add_player_to_map((row,col))
            self.current_coordin = (row, col)
        return self.current_coordin
    # -----------------------------
    def move_down(self):
        row, col = self.current_coordin
        row = row + 1
        if self.co.is_coordinates_within_vertical_range((row, col)):
            self.co.add_player_to_map((row, col))
            self.co.remove_player_prev_pos(self.current_coordin)
            self.current_coordin = (row, col)
        return self.current_coordin


