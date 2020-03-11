import numpy as np 
import pandas as pd
import sys
import os

from ase.atoms import Atom, Atoms
from ase.parallel import paropen

import seaborn as sns
sns.set()

def read_pdb(fileobj, index=-1):
    """Read PDB files.
    The format is assumed to follow the description given in
    http://www.wwpdb.org/documentation/format32/sect9.html."""
    if isinstance(fileobj, str):
        fileobj = open(fileobj)

    images = []
    atoms = Atoms()
    for line in fileobj.readlines():
        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                # Atom name is arbitrary and does not necessarily contain the element symbol.
                # The specification requires the element symbol to be in columns 77+78.
                symbol = line[76:78].strip().lower().capitalize()
                words = line[30:55].split()
                position = np.array([float(words[0]), 
                                     float(words[1]),
                                     float(words[2])])
                atoms.append(Atom(symbol, position))
            except:
                pass
        if line.startswith('ENDMDL'):
            images.append(atoms)
            atoms = Atoms()
    if len(images) == 0:
        images.append(atoms)
    return images[index]

from dscribe.descriptors import CoulombMatrix

dir_wt = './PDBs'
dir_mut = './database'
#dir_wt = './test_pdb'
#dir_mut = './test_mut'
files_wt_all = os.listdir(dir_wt)
files_wt_pdb = list(filter(lambda x: x.endswith('.pdb'), files_wt_all))
files_wt_hidden = list(filter(lambda x: x.startswith('.'), files_wt_pdb))
set1 = set(files_wt_pdb)
set2 = set(files_wt_hidden)
files_wt_pdb = list( set1.difference(set2) )

files_mut_all = os.listdir(dir_mut)

#distance_dict = {}
#Coulumb_data = pd.read_csv('data.csv')
pdb_list = [] #
dist_list = [] #

#Coulumb_data["Distance"] = ""
cm = CoulombMatrix(n_atoms_max = 8000, flatten=False, permutation = 'eigenspectrum')

i = 0
files_wt_pdb = ['1A4Y.pdb', '1A22.pdb'] #
for file_wt in files_wt_pdb:
	atoms_wt = read_pdb(dir_wt + '/' + file_wt)
	cm_wt = cm.create(atoms_wt)

	dist = []
	mutant_dict = {}	

	files_mut = filter(lambda x: x.startswith(file_wt[0:4]), files_mut_all)

	for file_mut in files_mut:
		print(file_mut)
		atoms = read_pdb(dir_mut + '/' + file_mut)
		cm_pdb = cm.create(atoms)	
		dist = np.linalg.norm(cm_pdb - cm_wt, ord = 2)
		#i = Coulumb_data[Coulumb_data['#Pdb'] == file_mut[0:-4]].index
		#Coulumb_data.at[i, 'Distance'] = dist
		dist_list.append(dist) #
		pdb_list.append(file_mut) #
		#mutant_dict[file_mut[5:-4]] =  dist
		#distance_dict[file_wt[5:-4]] =  mutant_dict'

#print(Coulumb_data)

Coulumb_data = pd.DataFrame({'#Pdb':pdb_list, 'Distance':dist_list}) #
Coulumb_data.to_csv("Coulumb_data.csv", index=False)

