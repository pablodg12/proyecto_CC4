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
    colors = ["red", "black", "green"]
  
    def __init__(self, pos, consumption=4, interphase_length=10):
        self.pos = pos # initial position
        self.state = 0 # inital state
        # Parameters
        self.consumption = consumption # resources consumed each timestep
        self.interphase_length = interphase_length # resources needed to enter mitosis
        self.size = np.random.randint(0, self.interphase_length//2) # initial size (consumed resources)

    def cycle(self, tumor, resources, nbohrs):
    	if self.state == 0:
    		if resources[self.pos] >= self.consumption:
    			resources[self.pos] -= self.consumption
    			self.size += self.consumption
    			if self.size >= self.interphase_length:
    				self.state = 2 # Go to Mitosis phase
    		else:
    			self.size += resources[self.pos]
    			resources[self.pos] = 0
    			self.state = 1 # Go to Resting phase

    	elif self.state == 1:
    		if self.size < self.interphase_length:
    			if resources[self.pos] >= self.consumption:
    				self.state = 0 # Go to interphase 
    		else:
    			nbohrs += self.pos
    			nbohrs[nbohrs < 0] = self.pos
    			nbohrs[nbohrs >= tumor.size] = self.pos
    			if np.any(tumor[nbohrs] == None):
    				self.state = 2
    	else:
    		nbohrs += self.pos
    		nbohrs[nbohrs < 0] = self.pos
    		nbohrs[nbohrs >= tumor.size] = self.pos
    		mask = (tumor[nbohrs] == None)*resources[nbohrs]
    		if not np.any(mask):
    			self.state = 1
    		else:
    			new_pos = nbohrs[np.argmax(mask)]
    			tumor[new_pos] = TumorCell(new_pos, consumption=self.consumption, interphase_length=self.interphase_length)
    			self.size = 0
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
        	consumption=params["tumor_cell_consumption"],
        	interphase_length=params["tumor_cell_interphase_length"])
        self.resources = [np.random.randint(params["resources_init_min"], params["resources_init_max"], self.N**2)]

    # Executes one timestep
    def timestep(self):
        self.resources.append(deepcopy(self.resources[-1]))
        self.tumor.append(deepcopy(self.tumor[-1]))

        self.resources_flux()
        self.tumor_timestep()

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
    	for i in range(0, self.N**2):
            if i < self.N or i > self.N**2 - self.N or i%self.N == 0 or i%self.N == self.N-1:
                continue
            nbohrs = self.get_tumor_nm(i) + i
            self.resources[-1][i] = self.resources[-2][i] + self.alpha/6*((self.resources[-2][nbohrs]).sum() - 6*self.resources[-2][i])

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