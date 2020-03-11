import sys
import os
import pandas as pd
from math import log10
from modeller import *
from modeller.optimizers import molecular_dynamics, conjugate_gradients
from modeller.automodel import autosched


aa = {"A":"ALA", "C":"CYS", "S":"SER", "T":"THR", "V":"VAL", "L":"LEU", "I":"ILE", "F":"PHE", "Y":"TYR", "W":"TRP", "H":"HIS", "G":"GLY", "E":"GLU", "D":"ASP", "Q":"GLN", "N":"ASN", "K":"LYS", "R":"ARG", "P":"PRO", "M":"MET"}
aa_inv = {"ALA":"A", "CYS":"C", "SER":"S", "THR":"T", "VAL":"V", "LEU":"L", "ILE":"I", "PHE":"F", "TYR":"Y", "TRP":"W", "HIS":"H", "GLY":"G", "GLU":"E", "ASP":"D", "GLN":"Q", "ASN":"N", "LYS":"K", "ARG":"R", "PRO":"P", "MET":"M"}

#
#  mutate_model.py
#
#     Usage:   python mutate_model.py modelname respos resname chain > logfile
#
#     Example: python mutate_model.py 1t29 1699 LEU A > 1t29.log
#
#
#  Creates a single in silico point mutation to sidechain type and at residue position
#  input by the user, in the structure whose file is modelname.pdb
#  The conformation of the mutant sidechain is optimized by conjugate gradient and
#  refined using some MD.
#
#  Note: if the model has no chain identifier, specify "" for the chain argument.
#


log.verbose()

# Set a different value for rand_seed to get a different final model
env = environ(rand_seed=-49837)

env.io.hetatm = True
env.io.water = True
env.io.atom_files_directory = ['./PDBs/']
#soft sphere potential
#lennard-jones potential (more accurate)
env.edat.dynamic_lennard=True
env.edat.contact_shell = 4.0
env.edat.update_dynamic = 0.39

# Read customized topology file with phosphoserines (or standard one)
env.libs.topology.read(file='$(LIB)/top_heav.lib')

# Read customized CHARMM parameter library with phosphoserines (or standard one)
env.libs.parameters.read(file='$(LIB)/par.lib')



def optimize(atmsel, sched):
    #conjugate gradient
    for step in sched:
        step.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)
    #md
    refine(atmsel)
    cg = conjugate_gradients()
    cg.optimize(atmsel, max_iterations=200, min_atom_shift=0.001)


#molecular dynamics
def refine(atmsel):
    # at T=1000, max_atom_shift for 4fs is cca 0.15 A.
    md = molecular_dynamics(cap_atom_shift=0.39, md_time_step=4.0,
                            md_return='FINAL')
    init_vel = True
    for (its, equil, temps) in ((200, 20, (150.0, 250.0, 400.0, 700.0, 1000.0)),
                                (200, 600,
                                 (1000.0, 800.0, 600.0, 500.0, 400.0, 300.0))):
        for temp in temps:
            md.optimize(atmsel, init_velocities=init_vel, temperature=temp,
                         max_iterations=its, equilibrate=equil)
            init_vel = False


#use homologs and dihedral library for dihedral angle restraints
def make_restraints(mdl1, aln):
   rsr = mdl1.restraints
   rsr.clear()
   s = selection(mdl1)
   for typ in ('stereo', 'phi-psi_binormal'):
       rsr.make(s, restraint_type=typ, aln=aln, spline_on_site=True)
   for typ in ('omega', 'chi1', 'chi2', 'chi3', 'chi4'):
       rsr.make(s, restraint_type=typ+'_dihedral', spline_range=4.0,
                spline_dx=0.3, spline_min_points = 5, aln=aln,
                spline_on_site=True)

#os.system("cp ./PDBs/" + modelname + ".pdb ./database/" + data["#Pdb"].iloc[i][0:4] + ".pdb")
data = pd.read_csv("skempi_v2.csv", sep=";")
pdb_list = []
p1_list = []
p2_list = []
Kd_list = []
t = []
more = []
#if (data["#Pdb"].iloc[i][0:4] == "3BT1") or (data["#Pdb"].iloc[i][0:4] == "1QHY") or (data["#Pdb"].iloc[i][0:4] == "1KBH"): continue
for i in range(len(data)):
    if (data["#Pdb"].iloc[i][0:4] == "3BT1") or (data["#Pdb"].iloc[i][0:4] == "1QHY") or (data["#Pdb"].iloc[i][0:4] == "3QHY") or (data["#Pdb"].iloc[i][0:4] == "1KBH"): continue
    print (data.iloc[i])
    modelname = data["#Pdb"].iloc[i][0:4] #+ "_rot"
    # copy wt pdb file and to the database folder and item to csv file
    database = os.listdir("./database")
    if data["#Pdb"].iloc[i][0:4] + ".pdb" in database:
        #os.system("cp database/" + data["#Pdb"].iloc[i][0:4] + ".pdb PDBs/" + modelname + ".pdb")
        os.system("cp PDBs/" + data["#Pdb"].iloc[i][0:4] + ".pdb database/" + data["#Pdb"].iloc[i][0:4] + "_rot"+ ".pdb")
        pdb, p1, p2 = data["#Pdb"].iloc[i].split("_")
        pdb_list.append(pdb)
        p1_list.append(p1)
        p2_list.append(p2)
        Kd_list.append(-log10(float(data["Affinity_mut_parsed"].iloc[i])))
        t.append(data["Temperature"][0:3])
        if data["Affinity_mut (M)"].iloc[i].startswith(">"):
            more.append(1)
        else:
            more.append(0)
        
    mut = data["Mutation(s)_cleaned"].iloc[i]
    respos = []
    restyp = []
    chain = []
    mut = mut.split(",")
    for m in mut:
        respos.append(m[2:-1])
        restyp.append(aa[m[-1]])
        chain.append(m[1])

    # Read the original PDB file and copy its sequence to the alignment array:
    mdl1 = model(env, file=modelname)
    ali = alignment(env)
    ali.append_model(mdl1, atom_files=modelname, align_codes=modelname)
     
    #set up the mutate residues selection segment
    for j in range(len(restyp)):
        s = selection(mdl1.chains[chain[j]].residues[respos[j]])
        #perform the mutate residue operation
        s.mutate(residue_type=restyp[j])


    #get two copies of the sequence.  A modeller trick to get things set up
    ali.append_model(mdl1, align_codes=modelname)

    # Generate molecular topology for mutant
    mdl1.clear_topology()
    mdl1.generate_topology(ali[-1])

    # Transfer all the coordinates you can from the template native structure
    # to the mutant (this works even if the order of atoms in the native PDB
    # file is not standard):
    #here we are generating the model by reading the template coordinates
    mdl1.transfer_xyz(ali)

    # Build the remaining unknown coordinates
    mdl1.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

    #yes model2 is the same file as model1.  It's a modeller trick.
    mdl2 = model(env, file=modelname)

    #required to do a transfer_res_numb
    #ali.append_model(mdl2, atom_files=modelname, align_codes=modelname)
    #transfers from "model 2" to "model 1"
    mdl1.res_num_from(mdl2, ali)

    #It is usually necessary to write the mutated sequence out and read it in
    #before proceeding, because not all sequence related information about MODEL
    #is changed by this command (e.g., internal coordinates, charges, and atom
    #types and radii are not updated).

    mdl1.write(file=modelname[0:4] + "_" + "".join([chain[j] + aa_inv[restyp[j]] + respos[j] for j in range(len(restyp))]) + '.tmp')
    mdl1.read(file=modelname[0:4] + "_" + "".join([chain[j] + aa_inv[restyp[j]] + respos[j] for j in range(len(restyp))]) + '.tmp')
    
    #set up restraints before computing energy
    #we do this a second time because the model has been written out and read in,
    #clearing the previously set restraints
    make_restraints(mdl1, ali)
    mdl1.patch_ss()

    #a non-bonded pair has to have at least as many selected atoms
    mdl1.env.edat.nonbonded_sel_atoms=1

    sched = autosched.loop.make_for_model(mdl1)

    #only optimize the selected residue (in first pass, just atoms in selected
    #residue, in second pass, include nonbonded neighboring atoms)
    #set up the mutate residue selection segment

    for j in range(len(restyp)):
        if j == 0:
            s = selection(mdl1.chains[chain[j]].residues[respos[j]])
            if str(int(respos[j]) - 1) in mdl1.residues: s.add(mdl1.chains[chain[j]].residues[str(int(respos[j]) - 1)])
            if str(int(respos[j]) + 1) in mdl1.residues: s.add(mdl1.chains[chain[j]].residues[str(int(respos[j]) + 1)]) 

        else:
            s.add(mdl1.chains[chain[j]].residues[respos[j]])
            if str(int(respos[j]) - 1) in mdl1.residues: s.add(mdl1.chains[chain[j]].residues[str(int(respos[j]) - 1)]) 
            if str(int(respos[j]) + 1) in mdl1.residues: s.add(mdl1.chains[chain[j]].residues[str(int(respos[j]) + 1)]) 
                

    mdl1.restraints.unpick_all()
    mdl1.restraints.pick(s)

    s.randomize_xyz(deviation=2.0)

    mdl1.env.edat.nonbonded_sel_atoms=2
    optimize(s, sched)

    #feels environment (energy computed on pairs that have at least one member
    #in the selected)
    mdl1.env.edat.nonbonded_sel_atoms=1
    optimize(s, sched)

    #give a proper name
    mdl1.write(file = "./database/" + modelname[0:4] + "_" + "".join([chain[j] + aa_inv[restyp[j]] + respos[j] for j in range(len(restyp))]) + '.pdb')

    #delete the temporary file
    os.remove(modelname[0:4] + "_" + "".join([chain[j] + aa_inv[restyp[j]] + respos[j] for j in range(len(restyp))]) + '.tmp')

    # append affinity information 
    pdb, p1, p2 = data["#Pdb"].iloc[i].split("_")
    pdb_list.append(modelname + "_" + "_".join([chain[j] + aa_inv[restyp[j]] + respos[j] for j in range(len(restyp))]))
    p1_list.append(p1)
    p2_list.append(p2)
    Kd_list.append(-log10(float(data["Affinity_mut_parsed"].iloc[i])))
    t.append(data["Temperature"][0:3])
    if data["Affinity_mut (M)"].iloc[i].startswith(">"):
        more.append(1)
    else:
        more.append(0)


final_data = pd.DataFrame({"#Pdb":pdb_list, "P1":p1_list, "P2":p2_list, "Pk":Kd_list, "Temperature":t, 'More':more})
final_data.to_csv("data.csv", index=False)






