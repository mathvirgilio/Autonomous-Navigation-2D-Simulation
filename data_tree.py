class Dados():
    def __init__(self, pos, possible_directions, route):
        self.pos = pos
        self.center = [(pos[0][0]+pos[-1][0])/2, (pos[0][1]+pos[-1][1])/2]
        self.possible_directions = possible_directions
        self.route = route   

class Node():
    def __init__(self, data, parent):
        self.data = data
        self.parent = parent
        self.up = None
        self.left = None
        self.right = None
        self.down = None

class data_tree():
    def __init__(self):
        self.root = Node(None, None)
        self.current_node = self.tree
        self.all_nodes = [self.root]
    def insert(self, data):
        new_node = Node(data, self.current_node)
        self.current_node = new_node
        self.all_nodes += [self.current_node]
    def search(self, pos):
        temp = self.root
        if(temp.data.pos == pos):
            return temp
        else:
    def list_nodes():
        temp = self.root
        if(temp.data.pos == pos):
            return temp
        else:



