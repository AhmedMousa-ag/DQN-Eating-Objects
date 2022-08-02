import numpy as np


class coordinates():
    def __init__(self, num_obj_per=0.20,height=100, width=100):
        # Defining our map structure which will be 100 dots for col and row

        self.col = width
        self.row = height
        # --------------------------------
        # Creating our map with Zeros which means nothing presented there
        self.map = np.zeros([self.col, self.row])
        self.num_obj_per = num_obj_per
        self.num_obj = int(self.num_obj_per * (self.col * self.row))
        self.object_coordinates = self.initate_obj()
    # ------------------------------------
    def update_objects_coordinates(self):
        self.object_coordinates = []
        for col in range(self.col):
            for row in range(self.row):
                if self.map[row, col] == 100:
                    self.object_coordinates.append((row, col))
        return self.object_coordinates

    # ------------------------------------
    def get_min_map(self):
        # This function for the environment only
        return np.zeros([self.col, self.row])
    # ---------------------------------------
    def get_obj_coord(self):
        return self.object_coordinates

    # ------------------------------------
    def get_map(self):
        scaled_map = self.map/ 255.
        return scaled_map
    # ------------------------------------
    def add_objects_to_map(self,coordinates):
        row,col = coordinates
        if self.is_coordinates_within_range(coordinates):
            self.map[row,col] = 100

        return self.update_objects_coordinates()
    # -------------------------------------
    def add_player_to_map(self,coordinates):
        row,col = coordinates
        if self.is_coordinates_within_range(coordinates):
            self.map[row,col] = 255
            self.remove_player_prev_pos(coordinates)
        return self.update_objects_coordinates()
    # -------------------------------------
    def remove_player_prev_pos(self,coordinates):
        row, col = coordinates
        if self.is_coordinates_within_range(coordinates):
            self.map[row, col] = 0
        return self.update_objects_coordinates()
    # -------------------------------------
    def remove_objects_from_map(self,coordinates):
        row,col = coordinates
        if self.is_coordinates_within_range(coordinates):
            print(f"Removed an Object")
            self.map[row,col] = 0
            self.num_obj = self.num_obj - 1
        return self.update_objects_coordinates()
    # ------------------------------------
    def is_pos_boundry(self,curr_pos):
        row,col = curr_pos
        if self.is_coordinates_within_range(curr_pos):
            if row == self.row-1 or col == self.col -1:
                return True
            else:
                return False
        else:
            False
    # ------------------------------------
    def rand_coord_gen(self,rand_seed):
        np.random.seed(rand_seed)
        row = np.random.randint(0, self.row)
        col = np.random.randint(0, self.col)
        return row,col
    # ------------------------------------
    # ------------------------------------
    def initate_obj(self):
        for i in range(self.num_obj):
            row,col = self.rand_coord_gen(i)
            self.add_objects_to_map((row,col))
   # ----------------------------------------
    def is_obj(self,coordinates):
        row,col = coordinates
        if self.is_coordinates_within_range(coordinates):
            co_val = self.map[row,col]
            if not co_val == 0.0:
                print(co_val)
            if co_val == 100:
                print("Found and object")
                return True
            else:
                return False
        else:
            return False
    # -----------------------------------------
    def is_coordinates_within_horizantel_range(self,coordinates):
        row, col = coordinates
        if row > self.row - 1:
            return False
        elif row < 0:
            return False
        elif col > self.col - 1:
            return False
        elif col < 0:
            return False
        else:
            return True
    # ---------------------------------------------------
    def is_coordinates_within_vertical_range(self,coordinates):
        row, col = coordinates
        if row > self.row - 1:
            return False
        elif row < 0:
            return False
        else:
            return True

    # --------------------

    def is_coordinates_within_range(self,coordinates):
        row,col = coordinates
        if col > self.col-1:
            return False
        elif col < 0:
            return False
        else:
            return True
    # -----------------------------------------------
    def reset(self):
        self.map = np.zeros([self.col, self.row])
        self.num_obj = int(self.num_obj_per * (self.col * self.row))
        self.initate_obj()
        self.object_coordinates = self.update_objects_coordinates()

