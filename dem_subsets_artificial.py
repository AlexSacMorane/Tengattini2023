#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

from yade import pack, plot, export
import numpy as np
import matplotlib.pyplot  as plt
import os, shutil, time, math, random, glob, pickle
from pathlib import Path

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 200
rMean = 0.000148 # -
rRelFuzz = 0.5
L_r = [] # initialization
density_grain = 2400 # kg/m3

# conversion pixel to µm
pixel_to_um_135 = 13.5 # µm/pixel
pixel_to_um_148 = 14.8 # µm/pixel
pixel_to_um = pixel_to_um_135
pixel_to_m = pixel_to_um*1e-6

# determine the domain dimension
Dz_on_Dx = 1 # ratio Dz / Dxy
Dz = 0.002 # m
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = 100

# grain properties 
density_grain = 2400 # kg/m3

# Mechanics Particle (TBD)
YoungModulus_particle = 4e9 # Pa
poisson_particle = 0.25 # -
alphaKrReal = 0.00
alphaKtwReal = alphaKrReal
frictionAngleReal = radians(15)

# Mechanics Bonds
YoungModulus_bond = YoungModulus_particle/5 # Pa 
# rupture (TBD)
tensileCohesion = 0.5*1e9 # Pa
shearCohesion = tensileCohesion # Pa
f_artificial = 10 # only for the IC

# Walls
P_confinement = 1.5e6 # Pa
P_cementation = 0.1*P_confinement
# triaxial 
v_wall_load = 1 # -/s
vert_strain_load = 0.1 # -
# controler
kp = 1e-9 # m.N-1
k_v_max = 0.000005 #-

# time step
factor_dt_crit = 0.2

# steady-state detection
window = 10
unbalancedForce_criteria = 0.01

# Report
simulation_report_name = O.tags['d.id']+'_report.txt'
simulation_report = open(simulation_report_name, 'w')
simulation_report.write('Triaxial Loading test\n')
simulation_report.write('Subset : artificial\n')
simulation_report.close()

#-------------------------------------------------------------------------------
#yade algorithm
#-------------------------------------------------------------------------------

O.engines = [
        PyRunner(command='grain_in_box()', iterPeriod = 1000),
        ForceResetter(),
        # sphere, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Wall_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere and sphere+wall
                # Ig : compute contact point. Ig2_Sphere (3DOF) or Ig2_Sphere6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Wall_Sphere_ScGeom()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(setCohesionNow=False, setCohesionOnNewContacts=False, label="physFunctor"), Ip2_FrictMat_FrictMat_FrictPhys()],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(always_use_moment_law=True, neverErase=True), Law2_ScGeom_FrictPhys_CundallStrack()]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.001, label = 'Newton'),
        PyRunner(command='checkUnbalanced_ir_ic()', iterPeriod = 200, label='checker')
]

#-------------------------------------------------------------------------------
#Initialisation
#-------------------------------------------------------------------------------

# clock to show performances
tic = time.perf_counter()
tic_0 = tic
iter_0 = 0

# plan simulation
os.mkdir('plot_'+O.tags['d.id'])
os.mkdir('data_'+O.tags['d.id'])

# define wall material (no friction)
O.materials.append(CohFrictMat(young=YoungModulus_particle, poisson=poisson_particle, frictionAngle=0, density=density_grain, isCohesive=False, momentRotationLaw=False))

# a list of 6 infinite walls enclosing the packing, in the order minX, maxX, minY, maxY, minZ, maxZ
O.bodies.append(wall(position=Vector3(    0, Dy/2, Dz/2), axis=0))
O.bodies.append(wall(position=Vector3(   Dx, Dy/2, Dz/2), axis=0))
O.bodies.append(wall(position=Vector3( Dx/2,    0, Dz/2), axis=1))
O.bodies.append(wall(position=Vector3( Dx/2,   Dy, Dz/2), axis=1))
O.bodies.append(wall(position=Vector3( Dx/2, Dy/2,    0), axis=2))
O.bodies.append(wall(position=Vector3( Dx/2, Dy/2,   Dz), axis=2))

# define grain material
O.materials.append(CohFrictMat(young=YoungModulus_particle, poisson=poisson_particle, frictionAngle=frictionAngleReal, density=density_grain,\
                               isCohesive=True, normalCohesion=tensileCohesion, shearCohesion=shearCohesion,\
                               momentRotationLaw=True, alphaKr=alphaKrReal, alphaKtw=alphaKtwReal))

# generate grain
for i in range(n_grains):
    radius = random.uniform(rMean*(1-rRelFuzz),rMean*(1+rRelFuzz))
    center_x = random.uniform(0+radius/n_steps_ic, Dx-radius/n_steps_ic)
    center_y = random.uniform(0+radius/n_steps_ic, Dy-radius/n_steps_ic)
    center_z = random.uniform(0+radius/n_steps_ic, Dz-radius/n_steps_ic)
    O.bodies.append(sphere(center=[center_x, center_y, center_z], radius=radius/n_steps_ic))
    # can use b.state.blockedDOFs = 'xyzXYZ' to block translation of rotation of a body
    L_r.append(radius)
O.tags['Step ic'] = '1'

# set the time step 
O.dt = factor_dt_crit*PWaveTimeStep()

#-------------------------------------------------------------------------------

# prepare the confinement step 
L_rel_error_x = []
L_rel_error_y = []
L_rel_error_z = []

#-------------------------------------------------------------------------------

def grain_in_box():
    '''
    Delete grains outside the box.
    '''
    #detect grain outside the box
    L_id_to_delete = []
    for b in O.bodies :
        if isinstance(b.shape, Sphere):
            #limit x \ limit y \ limit z
            if b.state.pos[0] < O.bodies[0].state.pos[0] or O.bodies[1].state.pos[0] < b.state.pos[0] or \
            b.state.pos[1] < O.bodies[2].state.pos[1] or O.bodies[3].state.pos[1] < b.state.pos[1] or \
            b.state.pos[2] < O.bodies[4].state.pos[2] or O.bodies[5].state.pos[2] < b.state.pos[2] :
                L_id_to_delete.append(b.id)
    if L_id_to_delete != []:
        #delete grain detected
        for id in L_id_to_delete:
            O.bodies.erase(id)
        #print and report
        simulation_report = open(simulation_report_name, 'a')
        simulation_report.write(str(len(L_id_to_delete))+" grains erased (outside of the box)\n")
        simulation_report.close()
        print("\nat ite "+str(O.iter)+': '+str(len(L_id_to_delete))+" grains erased (outside of the box)\n")

#-------------------------------------------------------------------------------

def count_bond():
    '''
    Count the number of bond.

    Correct Young modulus if necessary.
    '''
    counter_bond = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            # bond not broken
            if not i.phys.cohesionBroken :
                counter_bond = counter_bond + 1
            # correct Young modulus if bond broken 
            else :
                if i.phys.kn != YoungModulus_particle*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2):
                    localYoungModulus = YoungModulus_particle
                    i.phys.kn = localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2)
                    i.phys.ks = poisson_particle*localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2)
                    i.phys.kr = i.phys.ks*alphaKrReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
                    i.phys.ktw = i.phys.ks*alphaKtwReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
    return counter_bond


#-------------------------------------------------------------------------------

def checkUnbalanced_ir_ic():
    '''
    Increase particle radius until a steady-state is found.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
    # the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > unbalancedForce_criteria*5:
        return
    # increase the radius of particles
    if int(O.tags['Step ic']) < n_steps_ic :
        print('IC step '+O.tags['Step ic']+'/'+str(n_steps_ic)+' done')
        O.tags['Step ic'] = str(int(O.tags['Step ic'])+1)
        i_L_r = 0
        for b in O.bodies :
            if isinstance(b.shape, Sphere):
                growParticle(b.id, int(O.tags['Step ic'])/n_steps_ic*L_r[i_L_r]/b.shape.radius)
                i_L_r = i_L_r + 1
        # update the dt as the radii change
        O.dt = factor_dt_crit * PWaveTimeStep()
        return
    # characterize the ic algorithm
    global tic
    global iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nIC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('next step is the application of the cementation confinement\n')
    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced_load_cementation_ic()'
    checker.iterPeriod = 500
    # control top wall
    O.engines = O.engines + [PyRunner(command='controlWalls_ic()', iterPeriod = 1)]
    # switch on the gravity
    Newton.gravity = [0, 0, -9.81]
    # initialize trackers
    L_rel_error_x = []
    L_rel_error_y = []
    L_rel_error_z = []

#-------------------------------------------------------------------------------

def controlWalls_ic():
    '''
    Control the walls to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    # determine the maximum speed of the plate
    v_plate_max = rMean*k_v_max/O.dt
    # determine the center
    x_mean_dom = (O.bodies[0].state.pos[0]+O.bodies[1].state.pos[0])/2
    y_mean_dom = (O.bodies[2].state.pos[1]+O.bodies[3].state.pos[1])/2
    z_mean_dom = (O.bodies[4].state.pos[2]+O.bodies[5].state.pos[2])/2

    # x
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    if Fx == 0:
        O.bodies[0].state.pos =  (min([b.state.pos[0]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
        O.bodies[1].state.pos =  (max([b.state.pos[0]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
    else :
        dF = Fx - P_cementation*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[0].state.vel = (-np.sign(dF)*v_try_abs, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            O.bodies[0].state.vel = (-np.sign(dF)*v_plate_max, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_plate_max, 0, 0)

    # y
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    if Fy == 0:
        O.bodies[2].state.pos =  (x_mean_dom, min([b.state.pos[1]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
        O.bodies[3].state.pos =  (x_mean_dom, max([b.state.pos[1]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
    else :
        dF = Fy - P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_try_abs, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_plate_max, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_plate_max, 0)

    # z
    Fz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2
    if Fz == 0:
        O.bodies[4].state.pos =  (x_mean_dom, y_mean_dom, min([b.state.pos[2]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
        O.bodies[5].state.pos =  (x_mean_dom, y_mean_dom, max([b.state.pos[2]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            O.bodies[4].state.vel = (0, 0, -np.sign(dF)*v_try_abs)
            O.bodies[5].state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            O.bodies[4].state.vel = (0, 0, -np.sign(dF)*v_plate_max)
            O.bodies[5].state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def saveData_ic():
    """
    Save data in .txt file during the ic.
    """
    plot.saveDataTxt('data/IC_'+O.tags['d.id']+'.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_mean = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_ite  = []
    L_strain_x = []
    L_strain_y = []
    L_strain_z = []
    L_strain_vol = []
    L_n_bond = []
    file = 'data/IC_'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0]))
            L_sigma_y.append(abs(data[i][1]))
            L_sigma_z.append(abs(data[i][2]))
            L_sigma_mean.append((L_sigma_x[-1]+L_sigma_y[-1]+L_sigma_z[-1])/3)
            L_confinement.append(data[i][3])
            L_coordination.append(data[i][4])
            L_n_bond.append(data[i][5])
            L_ite.append(data[i][6])
            L_strain_x.append(data[i][8])
            L_strain_y.append(data[i][9])
            L_strain_z.append(data[i][10])
            L_strain_vol.append(L_strain_x[-1]+L_strain_y[-1]+L_strain_z[-1])
            L_unbalanced.append(data[i][11])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_y, label = r'$\sigma_y$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.plot(L_ite, L_sigma_mean, label = r'$\sigma_{mean}$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2.set_ylim(ymin=0, ymax=2*unbalancedForce_criteria)
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_ylim(ymin=0, ymax=150)
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_n_bond)
        ax3.set_title('Number of bond (-)')

        ax4.plot(L_ite, L_strain_x, label=r'$\epsilon_x$ (%)')
        ax4.plot(L_ite, L_strain_y, label=r'$\epsilon_y$ (%)')
        ax4.plot(L_ite, L_strain_z, label=r'$\epsilon_z$ (%)')
        ax4.legend()
        ax4.set_title('Strains (%)')

        ax5.plot(L_ite, L_strain_vol)
        ax5.set_title('Volumeric strain (%)')

        ax6.plot(L_ite, L_coordination)
        ax6.set_title('Coordination number (-)')

        plt.savefig('plot/IC_'+O.tags['d.id']+'.png')

        plt.close()

#-------------------------------------------------------------------------------

def checkUnbalanced_load_cementation_ic():
    '''
    Wait to reach the confining pressure targetted for cementation.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
    addPlotData_cementation_ic()
    saveData_ic()
    # compute targets
    target_x = P_cementation*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_y = P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_z = P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])
    # compute current measure
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    Fz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2
    # trackers
    L_rel_error_x.append(abs(Fx - target_x)/target_x)
    L_rel_error_y.append(abs(Fy - target_y)/target_y)
    L_rel_error_z.append(abs(Fz - target_z)/target_z)
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.01 or max(L_rel_error_y[-window:]) > 0.01 or max(L_rel_error_z[-window:]) > 0.01 :
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nPressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('next step is the activation of some parameters\n')
    # switch on friction, bending resistance and twisting resistance between particles
    O.materials[-1].frictionAngle = frictionAngleReal
    O.materials[-1].alphaKr = alphaKrReal
    O.materials[-1].alphaKtw = alphaKtwReal
    # for existing contacts, clear them
    O.interactions.clear()
    # calm down particles
    for b in O.bodies:
        if isinstance(b.shape,Sphere):
            b.state.angVel = Vector3(0,0,0)
            b.state.vel = Vector3(0,0,0)
    # switch off damping
    #Newton.damping = 0
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_param_ic()'
    # initialize trackers
    L_rel_error_x = []
    L_rel_error_y = []
    L_rel_error_z = []

#-------------------------------------------------------------------------------

def checkUnbalanced_param_ic():
    '''
    Wait to reach the equilibrium after switching on the friction and the rolling resistances.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
    addPlotData_cementation_ic()
    saveData_ic()    
    # compute targets
    target_x = P_cementation*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_y = P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_z = P_cementation*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])
    # compute current measure
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    Fz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2
    # trackers
    L_rel_error_x.append(abs(Fx - target_x)/target_x)
    L_rel_error_y.append(abs(Fy - target_y)/target_y)
    L_rel_error_z.append(abs(Fz - target_z)/target_z)
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.01 or max(L_rel_error_y[-window:]) > 0.01 or max(L_rel_error_z[-window:]) > 0.01 :
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Parameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.close()
    print("\nParameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('next step is the cementation\n')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'cementation()'
    checker.iterPeriod = 10
    # initialize trackers
    L_rel_error_x = []
    L_rel_error_y = []
    L_rel_error_z = []

#-------------------------------------------------------------------------------

def addPlotData_cementation_ic():
    """
    Save data in plot.
    """
    # add forces applied on walls
    sx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2/((O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=0,\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_cementation*100 + 1/3*sy/P_cementation*100 + 1/3*sz/P_cementation*100,\
                 strain_x=100*((O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0])-(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0]))/(O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0]),
                 strain_y=100*((O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1])-(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))/(O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1]),
                 strain_z=100*((O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2])-(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))/(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]))

#-------------------------------------------------------------------------------

def cementation():
    '''
    Generate cementation between grains.
    '''
    # generate the list of cohesive surface area and its list of weight
    cum_p_x_L, p_x_L, x_L = bsd_tengatini2023()
    # compute mean size of the cohesive surface
    mean_cohesiveSurface = np.average(x_L, weights=p_x_L)
    # counter
    global counter_bond0
    counter_bond0 = 0
    # iterate on interactions
    for i in O.interactions:
        # only grain-grain contact can be cemented
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere) :
            counter_bond0 = counter_bond0 + 1
            # creation of cohesion
            i.phys.cohesionBroken = False
            # determine the cohesive surface
            cohesiveSurface = random.choices(x_L, cum_weights=cum_p_x_L)[0]*1e-12 # m2
            # set normal and shear adhesions
            i.phys.normalAdhesion = tensileCohesion*cohesiveSurface
            i.phys.shearAdhesion = shearCohesion*cohesiveSurface
            # local law
            localYoungModulus = YoungModulus_particle + YoungModulus_bond*cohesiveSurface/mean_cohesiveSurface
            i.phys.kn = localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2)
            i.phys.ks = poisson_particle*localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2) 
            i.phys.kr = i.phys.ks*alphaKrReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
            i.phys.ktw = i.phys.ks*alphaKtwReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
    # write in the report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write(str(counter_bond0)+" contacts cemented initially\n\n")
    simulation_report.close()
    print('\n'+str(counter_bond0)+" contacts cemented")
    print('next step is the application of the loading confinement\n')

    # time step
    O.dt = factor_dt_crit * PWaveTimeStep()

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_load_confinement_ic()'
    checker.iterPeriod = 200
    # change the vertical pressure applied
    O.engines = O.engines[:-1] + [PyRunner(command='controlWalls_ic_b()', iterPeriod = 1)]

#-------------------------------------------------------------------------------

def bsd_tengatini2023():
    '''
    Define the weight of a bond size from (Tengattini, 2023).
    '''
    # reference
    L_ref_size     = [0, 0.182e4, 0.364e4, 0.546e4]
    L_ref_cum_prob = [0,    0.70,    0.90,       1]
    
    # input 
    size_min = 0.010e4 # µm2
    size_max = 0.546e4 # µm2
    n_size = 100
    L_size = np.linspace(size_min, size_max, n_size)

    # compute cumulative weight
    L_cum_p_size = []
    for size in L_size:
        # find interval
        i_ref_size = 0
        while not (L_ref_size[i_ref_size] <= size and size <= L_ref_size[i_ref_size+1]):
            i_ref_size = i_ref_size + 1
        # compute cumulative prob
        cum_p_size = L_ref_cum_prob[i_ref_size] + (L_ref_cum_prob[i_ref_size+1]-L_ref_cum_prob[i_ref_size])/(L_ref_size[i_ref_size+1]-L_ref_size[i_ref_size])*(size-L_ref_size[i_ref_size])
        L_cum_p_size.append(cum_p_size)

    # compute the weight
    L_p_size = []
    for i_size in range(len(L_size)):
        if i_size == 0:
            p_size = (L_cum_p_size[i_size+1]-L_cum_p_size[i_size])/(L_size[i_size+1]-L_size[i_size])
        if 0 < i_size and i_size < len(L_size)-1:
            p_size = (L_cum_p_size[i_size+1]-L_cum_p_size[i_size-1])/(L_size[i_size+1]-L_size[i_size-1])
        if i_size == len(L_size)-1:
            p_size = (L_cum_p_size[i_size]-L_cum_p_size[i_size-1])/(L_size[i_size]-L_size[i_size-1])
        L_p_size.append(p_size)
    
    # plot 
    fig, (ax1) = plt.subplots(1,1, figsize=(16,9),num=1)
    ax1.plot(L_size, L_cum_p_size)
    ax1.scatter(L_ref_size, L_ref_cum_prob, color='k')
    ax1.set_ylabel('cumulative percentage (-)')
    ax1.set_ylabel(r'bond size ($\mu m^2$)')
    fig.savefig('plot/BSD.png')
    plt.close()

    return L_cum_p_size, L_p_size, L_size

#-------------------------------------------------------------------------------

def controlWalls_ic_b():
    '''
    Control the walls to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    # determine the maximum speed of the plate
    v_plate_max = rMean*k_v_max/O.dt
    # determine the center
    x_mean_dom = (O.bodies[0].state.pos[0]+O.bodies[1].state.pos[0])/2
    y_mean_dom = (O.bodies[2].state.pos[1]+O.bodies[3].state.pos[1])/2
    z_mean_dom = (O.bodies[4].state.pos[2]+O.bodies[5].state.pos[2])/2

    # check the stress on x
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    if Fx == 0:
        O.bodies[0].state.pos =  (min([b.state.pos[0]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
        O.bodies[1].state.pos =  (max([b.state.pos[0]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
    else :
        dF = Fx - P_confinement*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[0].state.vel = (-np.sign(dF)*v_try_abs, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            O.bodies[0].state.vel = (-np.sign(dF)*v_plate_max, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_plate_max, 0, 0)
    
    # check the stress on y
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    if Fy == 0:
        O.bodies[2].state.pos =  (x_mean_dom, min([b.state.pos[1]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
        O.bodies[3].state.pos =  (x_mean_dom, max([b.state.pos[1]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
    else :
        dF = Fy - P_confinement*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_try_abs, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_plate_max, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_plate_max, 0)

    # check the stress on z
    Fz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2
    if Fz == 0:
        O.bodies[4].state.pos =  (x_mean_dom, y_mean_dom, min([b.state.pos[2]-0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
        O.bodies[5].state.pos =  (x_mean_dom, y_mean_dom, max([b.state.pos[2]+0.999*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_confinement*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            O.bodies[4].state.vel = (0, 0, -np.sign(dF)*v_try_abs)
            O.bodies[5].state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            O.bodies[4].state.vel = (0, 0, -np.sign(dF)*v_plate_max)
            O.bodies[5].state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_load_confinement_ic():
    '''
    Wait to reach the confining pressure.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
        
    # save and plot data
    SavePlot_data_confinement()
    
    # compute targets
    target_x = P_confinement*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_y = P_confinement*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
    target_z = P_confinement*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])
    # compute current measure
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    Fz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2
    # trackers
    L_rel_error_x.append(abs(Fx - target_x)/target_x)
    L_rel_error_y.append(abs(Fy - target_y)/target_y)
    L_rel_error_z.append(abs(Fz - target_z)/target_z)
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.01 or max(L_rel_error_y[-window:]) > 0.01 or max(L_rel_error_z[-window:]) > 0.01 :
        return
    if unbalancedForce() > unbalancedForce_criteria :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Confinining pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.close()
    print("\nConfining pressure applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('next step is the loading\n')

    # deactivate the artificial increase in bond strengths
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            # bond not broken
            if not i.phys.cohesionBroken :
                i.phys.normalAdhesion = i.phys.normalAdhesion/f_artificial
                i.phys.shearAdhesion = i.phys.shearAdhesion/f_artificial        

    # reset plot (IC done, simulation starts)
    plot.reset()

    # save new reference position for walls
    for id_wall in range(6):
        O.bodies[id_wall].state.refPos = O.bodies[id_wall].state.pos

    # triaxial loading
    O.engines = O.engines[:-1] + [PyRunner(command='controlWalls()', iterPeriod = 1)]
    
    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 1000

#-------------------------------------------------------------------------------

def SavePlot_data_confinement():
    """
    Save and plot data during the confinement.
    """
    # add forces applied on walls
    sx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2/((O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                 counter_bond=count_bond(), ratio_bond_broken=(counter_bond0-count_bond())/counter_bond0*100,\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_confinement*100 + 1/3*sy/P_confinement*100 + 1/3*sz/P_confinement*100,\
                 strain_x=100*((O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0])-(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0]))/(O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0]),
                 strain_y=100*((O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1])-(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))/(O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1]),
                 strain_z=100*((O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2])-(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))/(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]))

    # plot
    plot.saveDataTxt('data_'+O.tags['d.id']+'/confinement_'+O.tags['d.id']+'.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_mean = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_ite  = []
    L_strain_x = []
    L_strain_y = []
    L_strain_z = []
    L_strain_vol = []
    L_n_bond = []
    file = 'data_'+O.tags['d.id']+'/confinement_'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0])/1e6)
            L_sigma_y.append(abs(data[i][1])/1e6)
            L_sigma_z.append(abs(data[i][2])/1e6)
            L_sigma_mean.append((L_sigma_x[-1]+L_sigma_y[-1]+L_sigma_z[-1])/3)
            L_confinement.append(data[i][3])
            L_coordination.append(data[i][4])
            L_n_bond.append(data[i][5])
            L_ite.append(data[i][6])
            L_strain_x.append(data[i][9])
            L_strain_y.append(data[i][10])
            L_strain_z.append(data[i][11])
            L_strain_vol.append(L_strain_x[-1]+L_strain_y[-1]+L_strain_z[-1])
            L_unbalanced.append(data[i][12])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_y, label = r'$\sigma_y$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.plot(L_ite, L_sigma_mean, label = r'$\sigma_{mean}$')
        ax1.legend()
        ax1.set_title('Stresses (MPa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(ymin=0, ymax=2*unbalancedForce_criteria)
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.tick_params(axis='y', labelcolor='r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_ylim(ymin=0, ymax=150)
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_n_bond)
        ax3.set_title('Number of bond (-)')

        ax4.plot(L_ite, L_strain_x, label=r'$\epsilon_x$ (%)')
        ax4.plot(L_ite, L_strain_y, label=r'$\epsilon_y$ (%)')
        ax4.plot(L_ite, L_strain_z, label=r'$\epsilon_z$ (%)')
        ax4.legend()
        ax4.set_title('Strains (%)')

        ax5.plot(L_ite, L_strain_vol)
        ax5.set_title('Volumeric strain (%)')

        ax6.plot(L_ite, L_coordination)
        ax6.set_title('Coordination number (-)')

        plt.savefig('plot_'+O.tags['d.id']+'/confinement_'+O.tags['d.id']+'.png')

        plt.close()

#-------------------------------------------------------------------------------
#Load
#-------------------------------------------------------------------------------

def controlWalls():
    '''
    Control the walls to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    # determine the maximum speed of the plate
    v_plate_max = rMean*k_v_max/O.dt
    # determine the center
    x_mean_dom = (O.bodies[0].state.pos[0]+O.bodies[1].state.pos[0])/2
    y_mean_dom = (O.bodies[2].state.pos[1]+O.bodies[3].state.pos[1])/2
    z_mean_dom = (O.bodies[4].state.pos[2]+O.bodies[5].state.pos[2])/2

    # check the stress on x
    Fx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2
    if Fx == 0:
        O.bodies[0].state.pos =  (min([b.state.pos[0]-0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
        O.bodies[1].state.pos =  (max([b.state.pos[0]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), y_mean_dom, z_mean_dom)
    else :
        dF = Fx - P_confinement*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[0].state.vel = (-np.sign(dF)*v_try_abs, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            O.bodies[0].state.vel = (-np.sign(dF)*v_plate_max, 0, 0)
            O.bodies[1].state.vel = (np.sign(dF)*v_plate_max, 0, 0)
    
    # check the stress on y
    Fy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2
    if Fy == 0:
        O.bodies[2].state.pos =  (x_mean_dom, min([b.state.pos[1]-0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
        O.bodies[3].state.pos =  (x_mean_dom, max([b.state.pos[1]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), z_mean_dom)
    else :
        dF = Fy - P_confinement*(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_try_abs, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            O.bodies[2].state.vel = (0, -np.sign(dF)*v_plate_max, 0)
            O.bodies[3].state.vel = (0, np.sign(dF)*v_plate_max, 0)

    # apply load (speed control)
    O.bodies[4].state.vel = (0, 0,  v_wall_load*(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]))
    O.bodies[5].state.vel = (0, 0, -v_wall_load*(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]))

#-------------------------------------------------------------------------------

def checkUnbalanced():
    """
    Look for the equilibrium during the loading phase.
    """
    # save data
    SavePlot_data()

    # check simulation stop conditions
    if (O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2])-(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2])/(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]) > \
        vert_strain_load:
        stopLoad()
    
#-------------------------------------------------------------------------------

def stopLoad():
    """
    Close simulation.
    """
    # close yade
    O.pause()
    # characterize the dem step
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Triaxial loading test : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(count_bond())+" contacts cemented finally\n\n")
    simulation_report.close()
    print("Triaxial loading test : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('\n'+str(count_bond())+" contacts cemented\n")
    # characterize the last DEM step and the simulation
    hours = (tac-tic_0)//(60*60)
    minutes = (tac-tic_0 -hours*60*60)//(60)
    seconds = int(tac-tic_0 -hours*60*60 -minutes*60)
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.close()
    print("\nSimulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")

    # save simulation
    save_folder = 'Data_Tengattini2023/'+O.tags['d.id']
    os.mkdir(save_folder)
    shutil.move('data_'+O.tags['d.id'],save_folder+'/data')
    shutil.move('plot_'+O.tags['d.id'],save_folder+'/plot')
    shutil.move(O.tags['d.id']+'_report.txt', save_folder+'/'+O.tags['d.id']+'_report.txt')
    shutil.copy('dem_subsets.py',save_folder+'/dem_subsets.py')

#-------------------------------------------------------------------------------

def SavePlot_data():
    """
    Save data in plot.
    """
    # compute the stress on the three directions
    sx = (abs(O.forces.f(0)[0])+abs(O.forces.f(1)[0]))/2/((O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sy = (abs(O.forces.f(2)[1])+abs(O.forces.f(3)[1]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))
    sz = (abs(O.forces.f(4)[2])+abs(O.forces.f(5)[2]))/2/((O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0])*(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))
    
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                counter_bond=count_bond(), ratio_bond_broken=(counter_bond0-count_bond())/counter_bond0*100, bond_margin=compute_margin(),\
                Sx=sx, Sy=sy, Sz=sz, \
                conf_verified= 1/2*sx/(P_confinement)*100 + 1/2*sy/(P_confinement)*100, \
                strain_x=100*((O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0])-(O.bodies[1].state.pos[0]-O.bodies[0].state.pos[0]))/(O.bodies[1].state.refPos[0]-O.bodies[0].state.refPos[0]),
                strain_y=100*((O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1])-(O.bodies[3].state.pos[1]-O.bodies[2].state.pos[1]))/(O.bodies[3].state.refPos[1]-O.bodies[2].state.refPos[1]),
                strain_z=100*((O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2])-(O.bodies[5].state.pos[2]-O.bodies[4].state.pos[2]))/(O.bodies[5].state.refPos[2]-O.bodies[4].state.refPos[2]),
                x_min_dom=O.bodies[0].state.pos[0], x_max_dom=O.bodies[1].state.pos[0], y_min_dom=O.bodies[2].state.pos[1], y_max_dom=O.bodies[3].state.pos[1], z_min_dom=O.bodies[4].state.pos[2], z_max_dom=O.bodies[5].state.pos[2])

    # plot
    plot.saveDataTxt('data_'+O.tags['d.id']+'/'+O.tags['d.id']+'.txt')

    # post-proccess
    L_coordination = []
    L_n_bond = []
    L_margin_bond = []
    L_ratio_bond_broken = []
    L_ratio_bond_broken_pp = []
    L_unbalanced = []
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_deviatoric = []
    L_confinement = []
    L_strain_x = []
    L_strain_y = []
    L_strain_z = []
    L_shear_strain = []
    file = 'data_'+O.tags['d.id']+'/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:

        for i in range(len(data)):
            L_sigma_x.append(data[i][0]/1e6)
            L_sigma_y.append(data[i][1]/1e6)
            L_sigma_z.append(data[i][2]/1e6)
            L_sigma_deviatoric.append(1/2*(L_sigma_z[-1]-L_sigma_x[-1]) + 1/2*(L_sigma_z[-1]-L_sigma_y[-1]))
            L_margin_bond.append(data[i][3])
            L_confinement.append(data[i][4])
            L_coordination.append(data[i][5])
            L_n_bond.append(data[i][6])
            L_ratio_bond_broken_pp.append((data[0][6]-data[i][6])/data[0][6])
            L_ratio_bond_broken.append(data[i][9]/100)
            L_strain_x.append(abs(data[i][10]))
            L_strain_y.append(abs(data[i][11]))
            L_strain_z.append(abs(data[i][12]))
            L_shear_strain.append(abs(1/2*2/3*(L_strain_z[-1]-L_strain_x[-1]) + 1/2*2/3*(L_strain_z[-1]-L_strain_y[-1])))
            L_unbalanced.append(data[i][13])

        # Add Tengattini 2023 for 500, 1000, 1500 kPa of confinement
        # (8% cement)
        #L_strain_z_ref_500         = [0,  0.13,  1.03,   2.10,   2.85,   3.66,   4.82,   6.42,   8.11,   9.76,   11.0]
        #L_sigma_deviatoric_ref_500 = [0, 230e3, 678e3, 1166e3, 1423e3, 1449e3, 1399e3, 1336e3, 1248e3, 1250e3, 1200e3]
        #L_strain_z_ref_1000         = [0,  0.04,  0.29,   0.74,   1.32,   2.18,   3.32,   4.19,   5.04,   6.80,   8.05,   9.88,  11.48,  13.20]
        #L_sigma_deviatoric_ref_1000 = [0, 243e3, 742e3, 1434e3, 2138e3, 2778e3, 3163e3, 3356e3, 3292e3, 3038e3, 2732e3, 2568e3, 2352e3, 2276e3]
        L_strain_z_ref_1500         = [0,   0.85,   1.56,   2.43,   3.25,   4.90,   6.64,   8.20,   8.85,   9.74]
        L_sigma_deviatoric_ref_1500 = [0,  2.483,  3.545,  4.160,  4.532,  4.405,  4.100,  3.756,  3.513,  3.399] # MPa
        #L_strain_damage_ref_1500 = [0, 0.88, 2.00, 3.59, 5.03, 6.66, 8.23, 9.74]
        #L_damage_ref_1500        = [0, 0.07, 0.12, 0.19, 0.27, 0.34, 0.39, 0.44]
        
        # (6% cement)
        # not the ct-scan
        #L_strain_z_ref_1000         = [0,  0.21,   0.52,   0.81,   1.32,   1.84,   2.46,   2.96,   3.60,   4.20,   5.29,   5.98,   6.78,   7.48,   8.47,   9.52,   9.99]
        #L_sigma_deviatoric_ref_1000 = [0, 580e3, 1050e3, 1380e3, 1730e3, 2130e3, 2340e3, 2520e3, 2590e3, 2660e3, 2610e3, 2590e3, 2530e3, 2480e3, 2380e3, 2300e3, 2240e3]
        #L_strain_z_ref_1500         = [0,  0.13,   0.43,   0.78,   1.20,   1.89,   3.04,   4.05,   5.03,   6.34,   7.57,   8.85,   10.01]
        #L_sigma_deviatoric_ref_1500 = [0, 470e3, 1570e3, 2200e3, 2730e3, 3290e3, 3780e3, 4160e3, 4140e3, 3910e3, 3780e3, 3500e3,  3480e3]
        
        # plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

        ax1.plot(L_strain_z, L_coordination)
        ax1.set_ylabel('Coordination (-)')
        ax1.set_xlabel(r'$\epsilon_z$ (%)')

        ax2.plot(L_strain_z, L_n_bond, 'b')
        ax2.set_ylabel('Number (-)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_xlabel(r'$\epsilon_z$ (%)')
        ax2.set_title('Bonds (-)')
        ax2b = ax2.twinx()
        ax2b.plot(L_strain_z, L_ratio_bond_broken, 'r')
        ax2b.plot(L_strain_z, L_ratio_bond_broken_pp, 'indianred')
        ax2b.set_ylabel('Ratio (-)', color='r')
        ax2b.tick_params(axis='y', labelcolor='r')
        ax2c = ax2.twinx()
        ax2c.spines['right'].set_position(('outward', 60)) # offset
        ax2c.plot(L_strain_z, L_margin_bond, color='g')
        ax2c.set_ylabel('bond margin', color='g')
        ax2c.tick_params(axis='y', labelcolor='g')
        # add Tengattini 2023 
        #if P_confinement == 1.5e6:
        #    ax2b.plot(L_strain_damage_ref_1500, L_damage_ref_1500, linestyle='dashed', color='r')

        ax3.plot(L_strain_z, L_unbalanced, color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        ax3.set_ylabel('unbalanced (-)', color='b')
        ax3b = ax3.twinx()
        ax3b.plot(L_strain_z, L_confinement, color='r')
        ax3b.tick_params(axis='y', labelcolor='r')
        ax3b.set_ylabel('confinement (%)', color='r')
        ax3.set_xlabel(r'$\epsilon_z$ (%)')

        ax4.plot(L_strain_z, L_sigma_deviatoric)
        # add Tengattini 2023
        #if P_confinement == 0.5e6:
        #    ax4.plot(L_strain_z_ref_500, L_sigma_deviatoric_ref_500, linestyle='dashed', color='k')
        #if P_confinement == 1.0e6:
        #    ax4.plot(L_strain_z_ref_1000, L_sigma_deviatoric_ref_1000, linestyle='dashed', color='k')
        if P_confinement == 1.5e6:
            ax4.plot(L_strain_z_ref_1500, L_sigma_deviatoric_ref_1500, linestyle='dashed', color='k')
        ax4.set_xlabel(r'$\epsilon_z$ (%)')
        ax4.set_ylabel(r'Deviatoric stress (MPa)')
        
        #ax4.plot(L_shear_strain, L_sigma_deviatoric)
        #ax4.set_xlabel(r'$\epsilon_q$ (%)')
        #ax4.set_ylabel(r'Deviatoric stress (Pa)')

        #ax5.plot(L_strain_z, L_sigma_z)
        #ax5.set_xlabel(r'$\epsilon_z$ (%)')
        #ax5.set_ylabel(r'Vertical stress (Pa)')

        #ax6.plot(L_strain_z, L_strain_x, label=r'$\epsilon_x$')
        #ax6.plot(L_strain_z, L_strain_y, label=r'$\epsilon_y$')
        #ax6.legend(fontsize=15)
        #ax6.set_xlabel(r'$\epsilon_z$ (%)')
        #ax6.set_ylabel(r'Lateral strain (%)')

        plt.suptitle(r'Trackers - loading step (-)')
        plt.savefig('plot_'+O.tags['d.id']+'/'+O.tags['d.id']+'.png')
        plt.close()

#-------------------------------------------------------------------------------

def compute_margin():
    '''
    Compute the margin of the bonds before the rupture.

    The maximum between the tensile and shear is considered.
    '''
    # compute the margin for the bonds (force/stiffness) 
    margin_bond = 0
    counter_margin = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            if not i.phys.cohesionBroken :
                # tensile margin
                if i.geom.penetrationDepth < 0:
                    tensile_margin = np.linalg.norm(i.phys.normalForce)/i.phys.normalAdhesion
                else :
                    tensile_margin = 0
                # shear margin 
                shear_margin = np.linalg.norm(i.phys.shearForce)/(i.phys.shearAdhesion+i.phys.tangensOfFrictionAngle*np.linalg.norm(i.phys.normalForce))
                # take the maximum (largest potential to crack)
                margin_bond = margin_bond + max(tensile_margin, shear_margin)
                counter_margin = counter_margin + 1
    return margin_bond/counter_margin

#-------------------------------------------------------------------------------
# start simulation
#-------------------------------------------------------------------------------

O.run()
waitIfBatch()
