# -*- coding: utf-8 -*-
"""Proyecto_CC4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eYAoWTls03O9-YqtDaRp18URNwA-QK35

## Cancer Self-Remission and Tumor Stability
"""

import numpy as np
from abc import ABC
import matplotlib.pyplot as roberplot

# %matplotlib inline

# Base cell class
class Cell(ABC):
    def __init__(self, pos=0):
        self.pos = pos
   
    def plot(self, N, ax):
        x = 2*self.radius*(self.pos%N + ((self.pos//N)%2)/2)
        y = (self.pos//N)*np.sqrt(3)*self.radius
        circle = roberplot.Circle((x,y), self.radius, color=self.color, fill=False, lw=1.0, clip_on=False)
        ax.add_artist(circle)

# Tumor cell class

class TumorCell(Cell):
    radius = 2.0
    color = "red"
  
    def __init__(self, pos, parent):
        super().__init__(pos)
        self.parent = parent
  
    def reproduce(self, tissue, nm):
        # Choose birth place
        nm += self.pos
        if tissue.boundary != "rigid":
            nm[nm < 0] = nm[nm < 0]%tissue.tumor.size
            nm[nm >= tissue.tumor.size] = nm[nm >= tissue.tumor.size]%np.sqrt(tissue.tumor.size)
        else:
            nm[nm < 0] = self.pos
            nm[nm >= tissue.tumor.size] = self.pos
        mask = (tissue.tumor[nm] == None)*tissue.gm
        if not np.any(mask):
            return None
        new_pos = nm[np.argmax(np.random.rand(mask.size)*mask)]
        # Create cell 
        return TumorCell(new_pos, self.pos)

# T cell class

class TCell(Cell):
    radius = 1.0
    color = "red"

    def __init__(self, pos):
        super().__init__(pos)
    
    def move(self):
        return

# Tissue (grid) class
class Tissue():
    def __init__(self, N=10, init=0, boundary="rigid"):
        self.N = N # Lattice row size
        self.gm = np.random.choice([0.1,0.1,0.3,0.3,0.05,0.05], size=6, replace=False) # Irrigation gradient mask
        self.boundary = boundary
        self.tumor = np.empty(self.N**2, dtype=object) # Equilateral triangle lattice
        self.tumor[init] = TumorCell(init, -1)
        self.history = [self.tumor.copy()]
  
    # Returns the neighbors mask for a given cell with periodic boundary conditions.
    def get_nm_periodic(self, i):
        if (i%self.N == 0):
            if((i//self.N)%2 == 0):
                nm = [1, self.N, 2*self.N-1, self.N-1, -1, -self.N]
            else:
                nm = [1, self.N+1, self.N, self.N-1, -self.N, -self.N+1]
        elif (i%self.N == self.N-1):
            if(((i+1)//self.N)%2 == 0):
                nm = [-self.N+1, 1, self.N, -1, -self.N, -2*self.N+1]
            else:
                nm = [-self.N+1, self.N, self.N-1, -1, -self.N-1, -self.N]
        else:
            if((i//self.N)%2 == 0):
                nm = [1, self.N, self.N-1, -1, -self.N-1, -self.N]
            else:
                nm = [1, self.N+1, self.N, -1, -self.N, -self.N+1]
        return np.array(nm)

    # Returns the neighbors mask for a given cell with periodic boundary conditions.
    def get_nm_rigid(self, i):
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

    def tumor_centroid(self, timestep=-1):
        mx = (self.history[timestep] != None).reshape(self.N, self.N)
        x, y = np.meshgrid(np.arange(self.N), np.arange(self.N))
        xm = x[mx==1].mean()
        ym = y[mx==1].mean()
        pos = np.round(xm) + self.N*np.round(ym)
        return int(pos)

    def timestep(self):
        self.cancer_growth()
        self.history.append(self.tumor.copy())
  
    def cancer_growth(self):
        for cell in self.tumor[self.tumor != None]:
            if self.boundary=="rigid":
                nm = self.get_nm_rigid(cell.pos)
            else:
                nm = self.get_nm_periodic(cell.pos)
            new_cell = cell.reproduce(self, nm)
            if new_cell != None:
                self.tumor[new_cell.pos] = new_cell
  
    def plot_tumor(self, timestep=-1):
        fig, ax = roberplot.subplots(1)
        fig.set_size_inches(8,7.5)
        ax.set_xlim(-2, (self.N+2)*TumorCell.radius*2)
        ax.set_ylim(-2, (self.N+2)*TumorCell.radius*np.sqrt(3))
        for cell in self.history[timestep][self.history[timestep] != None]:
            cell.plot(self.N, ax)
            continue
        pos = self.tumor_centroid(timestep)
        x = 2*TumorCell.radius*(pos%self.N + ((pos//self.N)%2)/2)
        y = (pos//self.N)*np.sqrt(3)*TumorCell.radius
        circle = roberplot.Circle((x,y), TumorCell.radius, color=TumorCell.color, fill=True, lw=1.0, clip_on=False)
        ax.add_artist(circle)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cancerous tissue on timestep " + str(timestep))


