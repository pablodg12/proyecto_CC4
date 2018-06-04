import numpy as np
import matplotlib.pyplot as roberplot
import seaborn as sns
from copy import deepcopy

# Tumor cell class, with 3 possible states:
#	- Interphase: cell grows, sintetize DNA and prepares for mitosis (includes phases G1, S and G2).
#	- Mitosis: cell divides itself (phase M).
#	- Resting: cell does not divides itself and does not consume resources (phase G0).
class TumorCell():
    radius = 2.0
    colors = ["red", "orange", "green", "gray"]
  
    def __init__(self, pos, consumption=2, mitosis_threshold=10, apoptosis_threshold=10):
    	# Parameters
        self.consumption = consumption # resources consumed each timestep
        self.mitosis_threshold = mitosis_threshold # resources needed to enter mitosis
        self.apoptosis_threshold = apoptosis_threshold
        # Variables
        self.pos = pos # initial position
        self.state = 0 # inital state
        self.reserve = apoptosis_threshold
        self.size = np.random.randint(0, self.mitosis_threshold//2) # initial size (consumed resources)

    def cycle(self, tumor, resources, nbohrs):
    	if self.state != 3:
    		# Base consumption
    		if resources[self.pos] >= self.consumption:
    			resources[self.pos] -= self.consumption
    		elif self.reserve + resources[self.pos] >= self.consumption:
    			self.reserve -= (self.consumption - resources[self.pos])
    			resources[self.pos] = 0
    		else:
    			self.reserve = 0
    			self.state = 3
    			resources[self.pos] = 0
    			return
    		# State logic		
    		if self.state == 0:
    			if resources[self.pos] >= self.consumption:
    				resources[self.pos] -= self.consumption
    				self.size += self.consumption
    			elif self.reserve + resources[self.pos] >= self.consumption:
    				self.size += self.consumption
    				self.reserve -= (self.consumption - resources[self.pos])
    				resources[self.pos] = 0
    			else:
    				self.reserve = 0
    				self.state = 3
    				resources[self.pos] = 0
    				return
    			if self.size >= self.mitosis_threshold:
    				self.state = 2 # Go to Mitosis phase
    		elif self.state == 1:
    			nbohrs += self.pos
    			nbohrs[nbohrs < 0] = self.pos
    			nbohrs[nbohrs >= tumor.size] = self.pos
    			if np.any(tumor[nbohrs] == None):
    				self.state = 2
    		elif self.state == 2:
    			nbohrs += self.pos
    			nbohrs[nbohrs < 0] = self.pos
    			nbohrs[nbohrs >= tumor.size] = self.pos
    			mask = (tumor[nbohrs] == None)*resources[nbohrs]
    			if not np.any(mask):
    				self.state = 1
    			else:
    				new_pos = nbohrs[np.argmax(mask)]
    				tumor[new_pos] = TumorCell(new_pos, 
    					consumption=self.consumption, 
    					mitosis_threshold=self.mitosis_threshold, 
    					apoptosis_threshold=self.apoptosis_threshold)
    				self.size = np.random.randint(0, self.mitosis_threshold//2)
    				self.state = 0

    def plot(self, N):
        x = 2*self.radius*(self.pos%N + ((self.pos//N)%2)/2)
        y = (self.pos//N)*np.sqrt(3)*self.radius
        circle = roberplot.Circle((x,y), self.radius, color=self.colors[self.state], fill=False, lw=1.0, clip_on=False)
        return circle

# Tissue class, contains the honeycomb lattice with all the agents involved.
class Tissue():
    def __init__(self, params):
    	# Model parameters
        self.N = params["lattice_size"] # Lattice row length
        self.alpha = params["difussion_coef"] # Diffusion coefficient for resources flux.

        # Model structures
        self.tumor = [np.empty(self.N**2, dtype=object)]
        self.tumor[-1][params["tumor_init_pos"]] = TumorCell(params["tumor_init_pos"], 
        	consumption=params["tumorcell_consumption"],
        	mitosis_threshold=params["tumorcell_mitosis_threshold"],
        	apoptosis_threshold=params["tumorcell_apoptosis_threshold"])
        self.resources = [np.random.randint(params["resources_init_min"], params["resources_init_max"], self.N**2)]

    # Executes one timestep
    def timestep(self):
        self.resources.append(deepcopy(self.resources[-1]))
        self.tumor.append(deepcopy(self.tumor[-1]))

        self.tumor_timestep()
        self.resources_flux()

    # Returns the neighbors mask for a given cell with rigid boundary conditions.
    def get_tumor_nm(self, i):
        if (i%self.N == 0):
            if((i//self.N)%2 == 0):
                nm = [1, self.N, 0, 0, 0, -self.N]
            else:
                nm = [1, self.N+1, self.N, 0, -self.N, -self.N+1]
        elif (i%self.N == self.N-1):
            if(((i+1)//self.N)%2 == 0):
                nm = [0, 0, self.N, -1, -self.N, 0]
            else:
                nm = [0, self.N, self.N-1, -1, -self.N-1, -self.N]
        else:
            if((i//self.N)%2 == 0):
                nm = [1, self.N, self.N-1, -1, -self.N-1, -self.N]
            else:
                nm = [1, self.N+1, self.N, -1, -self.N, -self.N+1]
        return np.array(nm)

    def resources_flux(self):
    	aux = deepcopy(self.resources[-1])
    	for i in range(0, self.N**2):
            if i < self.N or i > self.N**2 - self.N or i%self.N == 0 or i%self.N == self.N-1:
                self.resources[-1][i] = self.resources[-2][i]
                continue
            nbohrs = self.get_tumor_nm(i) + i
            self.resources[-1][i] = aux[i] + self.alpha/6*((aux[nbohrs]).sum() - 6*aux[i])

    def tumor_timestep(self):
    	for cell in self.tumor[-1][self.tumor[-1] != None]:
    		cell.cycle(self.tumor[-1], self.resources[-1], self.get_tumor_nm(cell.pos))

    def plot_tissue(self, timestep=-1):
   		fig, ax = roberplot.subplots(1,2)
   		fig.set_size_inches(16,7.5)
   		ax[0].set_xlim(-2, (self.N+2)*TumorCell.radius*2)
   		ax[0].set_ylim(-2, (self.N+2)*TumorCell.radius*np.sqrt(3))
   		for cell in self.tumor[timestep][self.tumor[timestep] != None]:
   			ax[0].add_artist(cell.plot(self.N))
   			continue
   		ax[0].set_xlabel("x")
   		ax[0].set_ylabel("y")
   		ax[0].set_title("Tissue on timestep " + str(timestep))

   		sns.heatmap(self.resources[timestep].reshape(self.N, self.N), 
   			vmin=0, 
   			vmax=self.resources[0].max(), 
   			ax=ax[1],
   			xticklabels=False,
   			yticklabels=False)
   		ax[1].set_xlabel("x")
   		ax[1].set_ylabel("y")
   		ax[1].set_title("Resources on timestep " + str(timestep))

    def plot_data(self):
    	sums = []
    	for i in range(0, len(self.tumor)):
    		sums.append([0,0,0,0])
    		for cell in self.tumor[i][self.tumor[i] != None]:
    			sums[-1][cell.state] += 1

    	fig, ax = roberplot.subplots(1,2)
    	fig.set_size_inches(16,7.5)
    	[a,b,c,d] = ax[0].plot(sums)
    	ax[0].legend([a,b,c,d], ["Growing cells", "Resting cells", "Dividing Cells", "Dead cells"], loc=1)
    	ax[0].set_xlabel("Timestep")
    	ax[0].set_ylabel("N° of cells")
    	ax[0].set_title("Number of cells by state for each timestep")
    	ax[0].grid(True)

    	ax[1].scatter(np.arange(0, len(self.tumor)), np.log(np.sum(np.array(sums)[:,0:3], axis=1)), label="Active cells")
    	ax[1].scatter(np.arange(0, len(self.tumor)), np.log(np.sum(self.resources, axis=1)), label="Resources")
    	ax[1].set_xlabel("Timestep")
    	ax[1].set_ylabel("log Quantity")
    	ax[1].set_title("Number of active cells and resources")
    	ax[1].legend()
    	ax[1].grid(True)