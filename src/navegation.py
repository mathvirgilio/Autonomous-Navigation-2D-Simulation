from treelib import Tree
import numpy as np

class Dados():
    def __init__(self, pos, possible_directions, route):
        self.pos = pos
        self.center = [(pos[0][0]+pos[-1][0])/2, (pos[0][1]+pos[-1][1])/2]
        self.possible_directions = possible_directions
        self.route = route   

class Navegation():
    def __init__(self):
        self.data_tree = Tree()
        self.dead_end_found = False #Fim-da-linha, inicia o estado de retorno do robô
        self.end_simulation = False #Encerramento da simulação
        self.desired_position = None #Variável contendo o nome da posição atual a ser adicionada na árvore de dados
        self.parent = None
        self.current_directions = []
        self.previous_directions = []
        self.node_to_node_route = []
        self.orientation = 0
        
    def create_node(self, pos):
        node_id = str(pos[0])
        contains_position = self.data_tree.contains(node_id)
        
        if(contains_position):
            return False
        else:
            dados = Dados(np.copy(pos), self.current_directions, self.node_to_node_route)
            self.data_tree.create_node(node_id, node_id, data=dados, parent = self.parent)
            print("Adicionado nó", node_id, "Pai", self.parent)
            self.data_tree.get_node(node_id).data.possible_directions[self.orientation] = 2 #Marca a orientação atual como percorrida
            self.parent = node_id
            self.node_to_node_route = []
       
            return True
        
        return False
    
    def navegation_complete(self, pos):
        if(self.desired_position == str(pos)):
            #Se o estado de retorno é para encerrar a simulação...
            if(self.end_simulation):
                print("Fim da simulação")
                return True
                #pygame.quit()  
                #break
            
            print("De volta a", self.desired_position)
            self.dead_end_found = False
            #A partir da posição retornada, definir novo caminho para percorrer
            self.parent = str(self.desired_position)
            self.data_tree.get_node(self.desired_position).data.possible_directions[3-self.orientation] = 2
            self.current_directions = self.data_tree.get_node(self.desired_position).data.possible_directions
            
        return False
    
    def repeated_position(self, pos, orientation):
        possible_movement = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        move = possible_movement[orientation]
        new_pos = pos + move
        #print("Posição atual no teste", pos_)
        
        if(orientation == 0):
            new_pos[1] += 0.5
        elif(orientation == 1):
            new_pos[0] += 0.5
        elif(orientation == 2):
            new_pos[0] += 0.5
            new_pos[1] += 1
        elif(orientation == 3):
            new_pos[0] += 1
            new_pos[1] += 0.5
                
        for i in self.data_tree.all_nodes()[1:]:
            
            node = i.identifier
            parent = self.data_tree.parent(node).identifier
            [x1, y1] = self.data_tree.get_node(parent).data.center
            [x2, y2] = self.data_tree.get_node(node).data.center
            
            if(x1 > x2):
                x1, x2 = x2, x1
            if(y1 > y2):
                y1, y2 = y2, y1
            
            repeated_position = (new_pos[0] >= (x1 - 0.5)) and (new_pos[0] <= (x2+0.5))
            repeated_position = repeated_position and (new_pos[1] >= (y1-0.5)) and (new_pos[1] <= (y2+0.5))
                    
            if(repeated_position):
                return True
        
        return False
    
    def node_definition(self, pos, sucessive_moves):
        pos_label = pos[0]
        possible_movement = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        max_moves = 5
        valid_node = False
        
        #Primeira condição de criação de um nó -> Mudança de direção (direção atual não-disponível)'''
        if(self.direction_change()):
            valid_node = True #Validar a criação do nó
        #Segunda condição de criação de um nó -> Passar pelas condições de nó
        elif(self.node_condition()):
            valid_node = True
        
        robot_movement = possible_movement[self.orientation]
        
        #Verifica se o caminho escolhido a ser percorrido já foi passado antes
        #Terceira condição de criação de nó o caminho seguinte é trajetória já percorrida
        while(self.repeated_position(pos_label, self.orientation)):
            valid_node = True
            
            repeated_pos = str(pos_label+robot_movement) #Posição repetida
            print("Ponto repetido em", repeated_pos)
            
            self.current_directions[self.orientation] = 2
            contains_position = self.data_tree.contains(repeated_pos)
            if(contains_position):
                self.data_tree.get_node(repeated_pos).data.possible_directions[3-self.orientation] = 2 #Marca a orientação atual como percorrida
                        
            self.dead_end_check()
            if(self.dead_end_found):
                break
            else:
                robot_movement = possible_movement[self.orientation]
        
        print("Caminhos aqui", self.current_directions)
        
        #Caso tenha sido permitido a criação do nó nas condicionais anteriores
        if(valid_node or sucessive_moves > max_moves):
            node_created = self.create_node(pos) #Testa se a posição é repetida (False) ou não (True), criando o nó se não for
            return node_created
        
        return False
        
    def dead_end_check(self):
        self.dead_end_found = np.argwhere(self.current_directions==1).shape[0] == 0 #Se não houver caminhos detectados não-percorridos = True
        #Se não estiver em um dead-end: escolher nova orientação
        if(not self.dead_end_found):
            self.orientation = np.argwhere(self.current_directions==1)[0][0] #Orientação = Argumento da primeira direção disponível
    
    def direction_change(self):
        if(self.current_directions[self.orientation] != 1):
            #Detectar se há um dead-end:
            self.dead_end_check()
            #Se não houver um dead-end, escolher nova orientação
            if(not self.dead_end_found):
                #Orientação = Argumento da primeira direção disponível
                self.orientation = np.argwhere(self.current_directions==1)[0][0] 
            return True
        
        return False 
    
    def node_condition(self):
        condition1 = np.argwhere(self.current_directions==1).shape[0] > 1 #Possuir mais de um caminho
        condition2 = False #Possuir pelo menos um caminho diferente da última
        
        if(condition1):
            for i in range(len(self.current_directions)):
                if(self.current_directions[i] == 1 and self.previous_directions[i] == 0):
                    condition2 = True
        
        return condition1 and condition2
    
    def dead_end_procediment(self, pos):
        print("Fim da linha encontrado")
        self.desired_position = str(pos[0])
        
        root_found = self.data_tree.level(self.desired_position) == 0
        if(root_found):
            self.end_simulation = True
            aux = self.data_tree.get_node(self.desired_position).data.route
            aux.reverse()
            return_route = aux
        else:
            return_route = []
            while 1:
                #Definição da rota de retorno
                aux = self.data_tree.get_node(self.desired_position).data.route
                aux.reverse()              
                return_route.extend(aux) 
                #O pai vira nó atual na árvore de dados
                self.desired_position = self.data_tree.parent(self.desired_position).identifier
                print("Checando nó", self.desired_position)
                
                root_found = self.data_tree.level(self.desired_position) == 0 #Checar se a posição atual é a raiz
                new_directions = self.data_tree.get_node(self.desired_position).data.possible_directions
                
                for i in range(len(new_directions)):
                    if(new_directions[i] == 1):
                        orientation = i
                        if(self.repeated_position(self.data_tree.get_node(self.desired_position).data.pos[0], orientation)):
                            self.data_tree.get_node(self.desired_position).data.possible_directions[i] = 2
                            new_directions[i] = 2
                
                #Checar condição
                if np.argwhere(new_directions==1).shape[0] > 0:
                    print("Irá retornar para", self.desired_position)
                    break
                elif(root_found):
                    print("Irá encerrar a simulação em: ", self.desired_position)
                    self.end_simulation = True   
                    break
                
            if(len(return_route) > 2):
                short_cut_path = self.short_cut(str(pos[0]), self.desired_position)
                if(short_cut_path != []):
                    return_route = short_cut_path 
                    print("Há um atalho por:", short_cut_path)
                    
        return return_route
    
    def short_cut(self, begin, end):
        possible_movement = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        pos_begin = self.data_tree.get_node(begin).data.pos[0]
        pos_end = self.data_tree.get_node(end).data.pos[0]
        print("Procurar atalho entre:", pos_begin, pos_end)
        
        [x1, y1] = pos_begin
        [x2, y2] = pos_end
        
        print(x1, y1, x2, y2)
        
        if(x1 > x2):
            dir_x = 0
            x1, x2 = x2, x1
        else:
            dir_x = 3
        if(y1 > y2):
            dir_y = 1
            y1, y2 = y2, y1
        else:
            dir_y = 2
            
        hor_vert_path = (x2-x1)*[dir_x]+(y2-y1)*[dir_y]
        vert_hor_path = (y2-y1)*[dir_y]+(x2-x1)*[dir_x]
        
        hor_vert_path_possible = True
        vert_hor_path_path_possible = True
        pos_begin_cp = pos_begin
        
        for i, j in zip(hor_vert_path, vert_hor_path):
            robot_move = possible_movement[i]
            if(hor_vert_path_possible):
                if(self.repeated_position(pos_begin, i)):
                    pos_begin = pos_begin + robot_move
                else:
                    print("Falhou em:", pos_begin, i)
                    hor_vert_path_possible = False
            
            robot_move = possible_movement[j]
            
            if(vert_hor_path_path_possible):
                if(self.repeated_position(pos_begin_cp, j)):
                    pos_begin_cp = pos_begin_cp + robot_move
                else:
                    vert_hor_path_path_possible = False
            
            if(not hor_vert_path_possible and not vert_hor_path_path_possible):
                return []
        
        if(hor_vert_path_possible):
            return [3 - x for x in hor_vert_path]
        elif(vert_hor_path_path_possible):
            return [3 - x for x in vert_hor_path]
        else:
            return []