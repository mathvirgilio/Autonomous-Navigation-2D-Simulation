from operator import mod
from turtle import distance
import numpy as np
import pygame, sys
import simulador as sim
import navegation as nav
import math
import matplotlib.pyplot as plt

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GRAY = (100, 100, 100)
BACKGROUND = WHITE

class Dados():
    def __init__(self, pos, possible_directions, route):
        self.pos = pos
        self.center = [(pos[0][0]+pos[-1][0])/2, (pos[0][1]+pos[-1][1])/2]
        self.possible_directions = possible_directions
        self.route = route    

class simulation():
    def __init__(self):
        #Grid map
        self.map = None
        self.robot_pos = None
        #Simulation map
        self.screen_surf = None
        self.robot_surf = None
        self.robot_rect = None
        self.blocks = None
        self.blocks_surf = None
        self.image_surf = None
        self.distances = None
        
        #UVC measurements
        self.dosage_per_block = []
        self.irradiation_diagram = {}
        self.time_required = 1
        #Navegation algorithm
        self.navegation = None
        
    def read_map(self, file_name):
        #Ler arquivo e armazenar em vetor
        f = open(file_name) # opening a file
        map_file = f.read() # reading a file
        lines = []
        n_linhas = 1
        for x in map_file:
            if(x == '\n'):
                n_linhas = n_linhas+1
            else:
                lines.append(x)
        
        mapa = np.array(lines)
        self.map = mapa.reshape(n_linhas, len(lines)//n_linhas)
        f.close() # closing file object

    def fill_map(self, c):
        for i in self.robot_pos:
            x, y = i
            self.map[x, y] = c

    def create_robot(self, pos_inicial, robot_dim):
        robot_pos = []
        for i in range(robot_dim):
            for j in range(robot_dim):
                robot_pos.append([pos_inicial[0] + i, pos_inicial[1] + j])
        
        self.robot_pos = np.array(robot_pos) 
        self.robot_surf = pygame.Surface((50, 50))
        self.robot_rect = self.robot_surf.get_rect(center = (self.robot_pos[0][1]*25, self.robot_pos[0][0]*25))
        
    def move_robot(self, orientation):
        possible_movement = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        self.robot_pos += possible_movement[orientation]
   
    def create_display(self, file_name, initial_pos, robot_dim):

         self.read_map(file_name)
         self.create_robot(initial_pos, robot_dim)

         WINDOW_HEIGHT = self.map.shape[0]*25
         WINDOW_WIDTH = self.map.shape[1]*25
         pygame.init()
         self.screen_surf = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
         self.screen_surf.fill((255, 255, 255))
         self.blocks, self.blocks_surf = sim.initial_drawGrid(self.screen_surf, self.map)
         self.dosage_per_block = np.zeros((self.map.shape[1], self.map.shape[0]))
         self.image_surf = pygame.image.load("robot.png")
         self.image_surf = pygame.transform.scale(self.image_surf, (50, 50))
    
    def execute_navegation(self, power, necessary_dosage, attenuation, exposure_time):
        
        self.fill_map('x')
        self.navegation = nav.Navegation()
        self.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning')
        #Primeiro mapeamento
        current_directions_new = self.choose_way(self.mapping())
        self.navegation.current_directions = self.choose_way(self.mapping())
        #Primeiro nó criado
        self.navegation.orientation = np.argwhere(self.navegation.current_directions==1)[0][0] #Orientação = Argumento da primeira direção disponível
        self.navegation.create_node(self.robot_pos)
        #Primeiro movimento
        self.navegation.data_tree.get_node(str(self.robot_pos[0])).data.possible_directions[self.navegation.orientation] = 2
        self.node_to_node_route = [self.navegation.orientation]

        pygame.image.save(self.screen_surf, "inicial.jpeg")
        

        self.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='emitting')
        
        
        #print("Posição", self.robot_pos[0])
        #print("Orientação atual", self.navegation.orientation)
        #print("Caminhos aqui", self.navegation.current_directions)
        #print(self.map)
        
        iterations = 0
        return_route = []
        return_mode = False
        sucessive_moves = 0 #Nº de movimentos sucessivos sem nó criado
        
        test = True
        while 1:
            
            #Mudando a posição do robô no mapa
            self.fill_map('0')  #Preenche a posição anterior com '0'
            self.move_robot(self.navegation.orientation) #Executar movimento
            self.fill_map('x') #Preenche a nova posição com 'x'
            
            #print("Iteração", iterations)
            #print("Posição", self.robot_pos[0])
            #print("Orientação anterior", self.navegation.orientation)
            #print("Caminhos aqui", self.navegation.current_directions)  
            #print(self.map)
            
            #Simulador
            self.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='moving') #Atualiza o display
            if(not return_mode):
                self.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='scanning')
                self.show_raster()
            
            #Salvando a última leitura de escaneamento
            self.navegation.previous_directions = self.navegation.current_directions
            #Detectando caminhos possíveis em volta
            sensored_map = self.mapping()
            self.navegation.current_directions = self.choose_way(sensored_map)
            #self.show_raster()
            test = test and np.array_equal(current_directions_new, self.navegation.current_directions)
            #Registra a direção da onde o robô veio como percorrida (2)
            self.navegation.current_directions[3-self.navegation.orientation] = 2
            
            #Testa se a navegação deve ser encerrada.
            if(self.navegation.navegation_complete(self.robot_pos[0])):
                pygame.image.save(self.screen_surf, "screenshot.jpeg")
                pygame.quit()
                break
            #Se não estiver encontrado um dead-end (para qual procura o nó anterior)
            if(not self.navegation.dead_end_found): #if(len(return_route) == 0)
                #Testa se deve ser criado um nó na posição atual
                created_node = self.navegation.node_definition(self.robot_pos, sucessive_moves) #Função cria (True) ou não (False) o nó e retorna
                #Se o nó tiver sido criado, parar o robô por <exposure_time*time_delay> milisegundos para emitir UV-C no ponto:
                if(created_node):
                    sucessive_moves = 0 #Nº de movimentos sucessivos sem nó criado é zerado
                    self.show_navegation(power, necessary_dosage, exposure_time, attenuation, mode='emitting')
            
                #Se o nó encontrado for um dead-end, gerar trajetória de retorno
                if(self.navegation.dead_end_found): 
                    return_route = self.navegation.dead_end_procediment(self.robot_pos)
                #Caso contrário, salvar orientação atual e incrementar nº de movimentos sucessivos
                else:
                    self.navegation.node_to_node_route.append(self.navegation.orientation)
                    sucessive_moves += 1
                    
            #Testa se há trajetória de retorno (return_route != []), se sim, a orientação é modificada de acordo:
            return_mode = len(return_route) > 0
            if(return_mode):
                sucessive_moves = 0 #Nº de movimentos sucessivos sem nó mantido zerado
                self.navegation.orientation = 3 - return_route[0]
                #print("Rota de volta", return_route)
                return_route = return_route[1:] #Remove o movimento atual da lista de movimentos
                #print("Voltando para", self.navegation.desired_position, "onde", self.navegation.data_tree.get_node(self.navegation.desired_position).data.possible_directions)
            
            #self.time_required += 1 

            #print("Nova orientação", self.navegation.orientation)
            #print("Fim da iteração", iterations)
            #print("\n")
            iterations += 1
  
    def show_navegation(self, power, necessary_dosage, exposure_time, attenuation, mode, movement = 25, delta=np.zeros(3, dtype=int), pso=False):
        angle = [-180, -90, 90, 0]
        if(pso):
            orientation = 0
        else:
            orientation= self.navegation.orientation
        
        rotated_image = pygame.transform.rotate(self.image_surf, angle[orientation])
        if(mode == 'moving'):
            possible_movement = [[-1, 0], [0, -1], [0, 1], [1, 0]]
            v = 3
            for i in range(1, movement, v):

                self.screen_surf.fill(BACKGROUND) 
                self.robot_rect.x += v*possible_movement[self.navegation.orientation][1]
                self.robot_rect.y += v*possible_movement[self.navegation.orientation][0]
                #self.screen_surf.blit(self.robot_surf, self.robot_rect)
                self.drawGrid(power, necessary_dosage, attenuation, mode_emitting=True)
                
                self.screen_surf.blit(rotated_image, self.robot_rect)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                
                pygame.display.update()
                #pygame.time.delay(1) #Espera <time_delay> milisegundos
            
            self.robot_rect.x = self.robot_pos[0][1]*25
            self.robot_rect.y = self.robot_pos[0][0]*25
            #self.screen_surf.blit(self.robot_surf, self.robot_rect)
            self.drawGrid(power, necessary_dosage, attenuation)
            self.screen_surf.blit(rotated_image, self.robot_rect)
            self.time_required += 1
            pygame.display.update()

        elif(mode == 'scanning'):
            self.screen_surf.fill(BACKGROUND)  
            self.robot_rect.x = self.robot_pos[0][1]*25
            self.robot_rect.y = self.robot_pos[0][0]*25
            #self.screen_surf.blit(self.robot_surf, self.robot_rect)
            self.drawGrid(power, necessary_dosage, attenuation)
            
            self.screen_surf.blit(rotated_image, self.robot_rect)
            pygame.display.update()
            self.draw_radar(delta)
            self.time_required += 10
            #pygame.time.delay(1) #Espera <time_delay> milisegundos
            #pygame.image.save(self.screen_surf, "screenshot.jpeg")

        elif(mode == 'emitting'):
            self.robot_rect.x = self.robot_pos[0][1]*25
            self.robot_rect.y = self.robot_pos[0][0]*25
            for i in range(exposure_time):
                #self.screen_surf.blit(self.robot_surf, self.robot_rect)
                self.screen_surf.fill(BACKGROUND)  
                self.drawGrid(power, necessary_dosage, attenuation, mode_emitting=True)
                
                self.screen_surf.blit(rotated_image, self.robot_rect)
                self.time_required += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                #pygame.time.delay(1) #Espera <time_delay> milisegundos
                pygame.display.update()
                
    def drawGrid(self, power, necessary_dosage, attenuation, mode_emitting=False):
        mode_emitting = mode_emitting and (power != 0)
        navegation_map = self.map.T
        blockSize = 25 #Set the size of the grid block
        shape_x = navegation_map.shape[0]
        shape_y = navegation_map.shape[1]
        
        if(self.robot_rect.center not in self.irradiation_diagram.keys() and mode_emitting):
            self.irradiation_diagram[self.robot_rect.center] = []
            #self.irradiation_diagram[(self.robot_pos[0][0], self.robot_pos[0][1])] = []
        
        for i in range(shape_x):
            for j in range(shape_y):
                x = i*blockSize
                y = j*blockSize
                rect = pygame.Rect(x, y, blockSize, blockSize)

                if (navegation_map[i][j] != '1'):
                    dose = 0 
                    if(mode_emitting):
                        if(rect.center not in self.irradiation_diagram[self.robot_rect.center]):
                            if(sim.uv_irradiate(self.blocks, self.robot_rect, rect)):
                                self.irradiation_diagram[self.robot_rect.center] += [rect.center]
                                R = ( (self.robot_rect.center[0]-rect.center[0])**2+(self.robot_rect.center[1]-rect.center[1])**2)**(1/2)
                                dose = (power*attenuation/100)/(4*np.pi*(R**2))*1000
                        else:
                            R = ( (self.robot_rect.center[0]-rect.center[0])**2+(self.robot_rect.center[1]-rect.center[1])**2)**(1/2)
                            dose = (power*attenuation/100)/(4*np.pi*(R**2))*1000
                    
                    self.dosage_per_block[i][j] += dose
                    aux = int(255*self.dosage_per_block[i, j]/necessary_dosage)
                    fill = 1
                            
                    if(aux > 255):
                        aux = 255
                    fill = 0
                    
                    #pygame.draw.rect(self.screen_surf, (255,255-aux,255),rect, fill)
                    pygame.draw.rect(self.screen_surf, (255,255-aux,255),rect, fill)
                    pygame.draw.rect(self.screen_surf, BLACK,rect, 1)
                    font = pygame.font.SysFont(None, 12)
                    img = font.render(str(round(self.dosage_per_block[i][j], 1)), True, WHITE)
                    self.screen_surf.blit(img, (x+3, y+5))  
                else:
                    pygame.draw.rect(self.screen_surf, BLACK,rect, 0)
                    
    def draw_radar(self, delta=np.zeros(3, dtype=int)):
        x, y, theta = delta[0], delta[1], delta[2]
        xc, yc = self.robot_rect.center[0] + x, self.robot_rect.center[1] + y
        sensor_range = 100
        distances = []
        for angle in range(0 + theta, 360 + theta): 
            collide = False
            COLOR= RED
            for i in range(sensor_range):
                x = xc + i*math.sin(angle*np.pi/180)
                y = yc + i*math.cos(angle*np.pi/180)
                for block in self.blocks_surf:
                    collide = block.collidepoint(x, y)
                    if(collide):
                        break
                if(collide):
                    COLOR = BLUE
                    break
                else:
                    pass

            #pygame.draw.line(self.screen_surf, COLOR, self.robot_rect.center, (x, y), 1)
            d = np.sqrt((x-xc)**2+(y-yc)**2)
            distances.append(d)
            #pygame.time.delay(10)
            pygame.draw.circle(self.screen_surf, RED, (x, y), 1)

            pygame.display.update()
        
        self.distances = distances
    
    def mapping(self, raster_shape=(6,6), scale=25, depth = 1, delta=np.zeros(3)):
        raster = np.zeros(shape=raster_shape)
        i_center, j_center = (raster_shape[0])/2, (raster_shape[1])/2
        for angle, distance in enumerate(self.distances):
            if(distance <= 95):
                radians = angle*np.pi/180
                if(scale == 1):
                    displaciment = distance*np.math.sin(radians)
                    i = i_center + displaciment
                    displaciment = distance*np.math.cos(radians)
                    j = j_center + displaciment
                    if(np.any(delta)):
                        x, y, theta = delta
                        (i, j) = rotatePoint([i_center + x, j_center+y], [i + x, j + y], theta)
                    
                    i, j = round(i), round(j)
                else:
                    displaciment = (distance+scale)*np.math.sin(radians)//scale
                    i = i_center + displaciment
                    displaciment = (distance+scale)*np.math.cos(radians)//scale
                    j = j_center + displaciment
                    i, j = round(i), round(j)

                for di in range(-depth, depth + 1):
                    for dj in range(-depth, depth + 1):
                        if((i + di) >= 0 and (i + di) < raster_shape[0]):
                            if((j + dj) >= 0 and (j + dj) < raster_shape[1]):
                                raster[i+di][j+dj] = 1
            
        return np.transpose(raster)
    
    def choose_way(self, raster):
        #upway_available = not np.any(raster[0:2, 2:4])
        #leftway_available = not np.any(raster[2:4, 0:2])
        #rightway_available = not np.any(raster[2:4, 4:6])
        #downway_available = not np.any(raster[4:6, 2:4])
        upway_available = not np.any(raster[1, 2:4])
        leftway_available = not np.any(raster[2:4, 1])
        rightway_available = not np.any(raster[2:4, 4])
        downway_available = not np.any(raster[4, 2:4])
        return np.array([upway_available, leftway_available, rightway_available, downway_available], dtype=int)
    
    def show_raster(self):
        sensored_map = self.mapping(raster_shape=(200,200), scale=1, depth = 2)
        #plt.imshow(sensored_map,cmap='gray_r')
        #plt.show()
        #sensored_map = self.mapping()
        plt.imshow(sensored_map,cmap='gray_r')
        plt.show()

def rotatePoint(centerPoint,point,angle):
    """Rotates a point around another centerPoint. Angle is in degrees.
    Rotation is counter-clockwise"""
    angle = math.radians(angle)
    temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1] #mudar sinal
    temp_point = ( temp_point[0]*math.cos(angle)-temp_point[1]*math.sin(angle) , temp_point[0]*math.sin(angle)+temp_point[1]*math.cos(angle))
    temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
    return temp_point
