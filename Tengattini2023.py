#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

from yade import pack, plot, export
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time
import math
import random
import pickle
from pathlib import Path

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 3000
rMean = 0.000275/2  # m
rRelFuzz = 0.449
L_r = [] # initialization
density_grain = 2400 # kg/m3

# Box
Dz_on_Dx = 1 # ratio Dz / Dxy
Dz = 0.004 # m
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = 100

# Walls
P_confinement = 1e6 # Pa
# triaxial 
vert_strain_load = 0.1 # -
n_load = 20
# controler
kp = 2*1e-10 # m.N-1
k_v_max = 0.000005 #-

# cementation
P_cementation = P_confinement*1 # Pa

# Mechanics Particle (TBD)
YoungModulus_particle = 1e9 # Pa
poisson_particle = 0.25 # -
alphaKrReal = 0.5
alphaKtwReal = 0.5
frictionAngleReal = radians(20)

# Mechanics Bonds
YoungModulus_bond = YoungModulus_particle/5 # Pa 
# rupture (TBD)
tensileCohesion = 5e6 # Pa
shearCohesion = tensileCohesion*2.4 # Pa

# time step
factor_dt_crit_1 = 0.6
factor_dt_crit_2 = 0.2

# steady-state detection
window = 10
unbalancedForce_criteria = 0.1

# Report
simulation_report_name = O.tags['d.id']+'_report.txt'
simulation_report = open(simulation_report_name, 'w')
simulation_report.write('Triaxial Loading test\n')
simulation_report.write('Type of sample: Rock\n')
simulation_report.write('Cementation at '+str(int(P_cementation))+' Pa\n')
simulation_report.write('Confinement at '+str(int(P_confinement))+' Pa\n\n')
simulation_report.close()

#-------------------------------------------------------------------------------
#Initialisation
#-------------------------------------------------------------------------------

# clock to show performances
tic = time.perf_counter()
tic_0 = tic
iter_0 = 0

# plan simulation
if Path('plot').exists():
    shutil.rmtree('plot')
os.mkdir('plot')
if Path('data').exists():
    shutil.rmtree('data')
os.mkdir('data')
if Path('vtk').exists():
    shutil.rmtree('vtk')
os.mkdir('vtk')
if Path('save').exists():
    shutil.rmtree('save')
os.mkdir('save')

# define wall material (no friction)
O.materials.append(CohFrictMat(young=YoungModulus_particle, poisson=poisson_particle, frictionAngle=0, density=density_grain, isCohesive=False, momentRotationLaw=False))

# create box and grains
O.bodies.append(aabbWalls([Vector3(0,0,0),Vector3(Dx,Dy,Dz)], thickness=0.,oversizeFactor=1))
# a list of 6 boxes Bodies enclosing the packing, in the order minX, maxX, minY, maxY, minZ, maxZ
# extent the plates
O.bodies[0].shape.extents = Vector3(0,1.5*Dy/2,1.5*Dz/2)
O.bodies[1].shape.extents = Vector3(0,1.5*Dy/2,1.5*Dz/2)
O.bodies[2].shape.extents = Vector3(1.5*Dx/2,0,1.5*Dz/2)
O.bodies[3].shape.extents = Vector3(1.5*Dx/2,0,1.5*Dz/2)
O.bodies[4].shape.extents = Vector3(1.5*Dx/2,1.5*Dy/2,0)
O.bodies[5].shape.extents = Vector3(1.5*Dx/2,1.5*Dy/2,0)
# global names
plate_x = O.bodies[1]
plate_y = O.bodies[3]
plate_z = O.bodies[5]

# define grain material
# frictionAngle, alphaKr, alphaKtw are set to 0 during IC. The real value is set after IC.
O.materials.append(CohFrictMat(young=YoungModulus_particle, poisson=poisson_particle, frictionAngle=atan(0.05), density=density_grain,\
                               isCohesive=True, normalCohesion=tensileCohesion, shearCohesion=shearCohesion,\
                               momentRotationLaw=True, alphaKr=0, alphaKtw=0))

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

# yade algorithm
O.engines = [
        PyRunner(command='grain_in_box()', iterPeriod = 1000),
        ForceResetter(),
        # sphere, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere and sphere+wall
                # Ig : compute contact point. Ig2_Sphere (3DOF) or Ig2_Sphere6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Box_Sphere_ScGeom6D()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(always_use_moment_law=True)]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.001, label = 'Newton'),
        PyRunner(command='checkUnbalanced_ir_ic()', iterPeriod = 200, label='checker')
]
# time step
O.dt = factor_dt_crit_1 * PWaveTimeStep()

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
        print("\n"+str(len(L_id_to_delete))+" grains erased (outside of the box)\n")

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_ic():
    '''
    Increase particle radius until a steady-state is found.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
    # the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > .1:
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
        O.dt = factor_dt_crit_1 * PWaveTimeStep()
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
    # save
    #O.save('save/simu_ic.yade.bz2')
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
    Fx = O.forces.f(plate_x.id)[0]
    if Fx == 0:
        plate_x.state.pos =  (max([b.state.pos[0]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_y.state.pos[1]/2, plate_z.state.pos[2]/2)
    else :
        dF = Fx - P_cementation*plate_y.state.pos[1]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_x.state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            plate_x.state.vel = (np.sign(dF)*v_plate_max, 0, 0)
    
    Fy = O.forces.f(plate_y.id)[1]
    if Fy == 0:
        plate_y.state.pos =  (plate_x.state.pos[0]/2, max([b.state.pos[1]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_z.state.pos[2]/2)
    else :
        dF = Fy - P_cementation*plate_x.state.pos[0]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_y.state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            plate_y.state.vel = (0, np.sign(dF)*v_plate_max, 0)

    Fz = O.forces.f(plate_z.id)[2]
    if Fz == 0:
        plate_z.state.pos =  (plate_x.state.pos[0]/2, plate_y.state.pos[1]/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_cementation*plate_x.state.pos[0]*plate_y.state.pos[1]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            plate_z.state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            plate_z.state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_load_cementation_ic():
    '''
    Wait to reach the confining pressure targetted for cementation.
    '''
    global L_rel_error_x, L_rel_error_y, L_rel_error_z
    addPlotData_cementation_ic()
    saveData_ic()
    # trackers
    L_rel_error_x.append(abs(O.forces.f(plate_x.id)[0] - P_cementation*plate_y.state.pos[1]*plate_z.state.pos[2])/(P_cementation*plate_y.state.pos[1]*plate_z.state.pos[2]))
    L_rel_error_y.append(abs(O.forces.f(plate_y.id)[1] - P_cementation*plate_x.state.pos[0]*plate_z.state.pos[2])/(P_cementation*plate_x.state.pos[0]*plate_z.state.pos[2]))
    L_rel_error_z.append(abs(O.forces.f(plate_z.id)[2] - P_cementation*plate_x.state.pos[0]*plate_y.state.pos[1])/(P_cementation*plate_x.state.pos[0]*plate_y.state.pos[1]))
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.005 or max(L_rel_error_y[-window:]) > 0.005 or max(L_rel_error_z[-window:]) > 0.005 :
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
    # trackers
    L_rel_error_x.append(abs(O.forces.f(plate_x.id)[0] - P_cementation*plate_y.state.pos[1]*plate_z.state.pos[2])/(P_cementation*plate_y.state.pos[1]*plate_z.state.pos[2]))
    L_rel_error_y.append(abs(O.forces.f(plate_y.id)[1] - P_cementation*plate_x.state.pos[0]*plate_z.state.pos[2])/(P_cementation*plate_x.state.pos[0]*plate_z.state.pos[2]))
    L_rel_error_z.append(abs(O.forces.f(plate_z.id)[2] - P_cementation*plate_x.state.pos[0]*plate_y.state.pos[1])/(P_cementation*plate_x.state.pos[0]*plate_y.state.pos[1]))
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.005 or max(L_rel_error_y[-window:]) > 0.005 or max(L_rel_error_z[-window:]) > 0.005 :
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
    # save
    #O.save('save/'+O.tags['d.id']+'_ic.yade.bz2')
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'cementation()'
    checker.iterPeriod = 10
    # initialize trackers
    L_rel_error_x = []
    L_rel_error_y = []
    L_rel_error_z = []

#-------------------------------------------------------------------------------

def cementation():
    '''
    Generate cementation between grains.
    '''
    # generate the list of cohesive surface area and its list of weight
    x_min = 0.13e4 # µm2
    x_max = 3.20e4 # µm2
    n_x = 200
    x_L = np.linspace(x_min, x_max, n_x)
    p_x_L = bsd_tenngatini2023(x_L)
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
            cohesiveSurface = random.choices(x_L,p_x_L)[0]*1e-12 # m2
            # set normal and shear adhesions
            i.phys.normalAdhesion = tensileCohesion*cohesiveSurface
            i.phys.shearAdhesion = shearCohesion*cohesiveSurface
            # average law
            localYoungModulus = YoungModulus_particle+YoungModulus_bond
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
    O.dt = factor_dt_crit_2 * PWaveTimeStep()

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_load_confinement_ic()'
    checker.iterPeriod = 200
    # change the vertical pressure applied
    O.engines = O.engines[:-1] + [PyRunner(command='controlWalls_ic_b()', iterPeriod = 1)]

#-------------------------------------------------------------------------------

def bsd_tenngatini2023(L_size):
    '''
    Define the weight of a bond size from (Tengattini, 2023).
    '''
    # reference (8% cement)
    L_ref_size = [0.13e4, 0.28e4, 0.51e4, 0.72e4, 0.91e4, 1.14e4, 1.37e4, 1.65e4, 2.01e4, 2.36e4, 2.77e4, 3.20e4]
    L_ref_prob = [  0.02,   0.04,   0.12,   0.16,   0.13,   0.14,   0.12,   0.11,   0.07,   0.04,   0.02,   0.03]
    L_ref_cum_prob = [L_ref_prob[0]]
    for ref_prob in L_ref_prob[1:]:
        L_ref_cum_prob.append(L_ref_cum_prob[-1] + ref_prob)

    # compute weight
    L_p_size = []
    for size in L_size:
        # out of range
        if size < L_ref_size[0] or L_ref_size[-1] < size :
            L_p_size.append(0)
        # in range
        else :
            # find interval
            i_ref_size = 0
            while not (L_ref_size[i_ref_size] <= size and size <= L_ref_size[i_ref_size+1]):
                i_ref_size = i_ref_size + 1
            # compute weight
            p_size = L_ref_prob[i_ref_size] + (L_ref_prob[i_ref_size+1]-L_ref_prob[i_ref_size])/(L_ref_size[i_ref_size+1]-L_ref_size[i_ref_size])*(size-L_ref_size[i_ref_size])
            L_p_size.append(p_size)
    # normalize
    sum_p_size = np.sum(L_p_size)
    for i_p_size in range(len(L_p_size)):
        L_p_size[i_p_size] = L_p_size[i_p_size]/sum_p_size
    # compute cumulative
    L_cum_p_size = [L_p_size[0]]
    for p_size in L_p_size[1:]:
        L_cum_p_size.append(L_cum_p_size[-1]+p_size)
    
    # plot 
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,9),num=1)
    # percentage
    ax1.plot(L_size, L_p_size)
    ax1.set_ylabel('percentage (-)')
    # cumulative
    ax2.plot(L_size, L_cum_p_size)
    ax2.plot(L_ref_size, L_ref_cum_prob, linestyle='dashed', color='k')
    ax2.set_ylabel('cumulative percentage (-)')
    ax2.set_ylabel(r'bond size ($\mu m^2$)')
    # close
    fig.savefig('plot/BSD.png')
    plt.close()

    return L_p_size

#-------------------------------------------------------------------------------

def controlWalls_ic_b():
    '''
    Control the walls to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fx = O.forces.f(plate_x.id)[0]
    if Fx == 0:
        plate_x.state.pos =  (max([b.state.pos[0]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_y.state.pos[1]/2, plate_z.state.pos[2]/2)
    else :
        dF = Fx - P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_x.state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            plate_x.state.vel = (np.sign(dF)*v_plate_max, 0, 0)
    
    Fy = O.forces.f(plate_y.id)[1]
    if Fy == 0:
        plate_y.state.pos =  (plate_x.state.pos[0]/2, max([b.state.pos[1]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_z.state.pos[2]/2)
    else :
        dF = Fy - P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_y.state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            plate_y.state.vel = (0, np.sign(dF)*v_plate_max, 0)

    Fz = O.forces.f(plate_z.id)[2]
    if Fz == 0:
        plate_z.state.pos =  (plate_x.state.pos[0]/2, plate_y.state.pos[1]/2, max([b.state.pos[2]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_confinement*plate_x.state.pos[0]*plate_y.state.pos[1]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            plate_z.state.vel = (0, 0, np.sign(dF)*v_try_abs)
        else :
            plate_z.state.vel = (0, 0, np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_load_confinement_ic():
    '''
    Wait to reach the vertical/lateral pressure targetted for confinement.
    '''
    global i_load, L_unbalanced_ite, L_confinement_x_ite, L_confinement_y_ite, L_count_bond, vert_pos_load, L_rel_error_x, L_rel_error_y, L_rel_error_z 
    addPlotData_confinement_ic()
    saveData_ic()
    # trackers
    L_rel_error_x.append(abs(O.forces.f(plate_x.id)[0] - P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2])/(P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2]))
    L_rel_error_y.append(abs(O.forces.f(plate_y.id)[1] - P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2])/(P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2]))
    L_rel_error_z.append(abs(O.forces.f(plate_z.id)[2] - P_confinement*plate_x.state.pos[0]*plate_y.state.pos[1])/(P_confinement*plate_x.state.pos[0]*plate_y.state.pos[1]))
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.005 or max(L_rel_error_y[-window:]) > 0.005 or max(L_rel_error_z[-window:]) > 0.005 :
        return
    # characterize the ic algorithm
    global tic, iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac

    # compute mean overlap/diameter
    L_over_diam = []
    for contact in O.interactions:
        if isinstance(O.bodies[contact.id1].shape, Sphere) and isinstance(O.bodies[contact.id2].shape, Sphere):
            b1_x = O.bodies[contact.id1].state.pos[0]
            b1_y = O.bodies[contact.id1].state.pos[1]
            b1_z = O.bodies[contact.id1].state.pos[2]
            b2_x = O.bodies[contact.id2].state.pos[0]
            b2_y = O.bodies[contact.id2].state.pos[1]
            b2_z = O.bodies[contact.id2].state.pos[2]
            dist = math.sqrt((b1_x-b2_x)**2+(b1_y-b2_y)**2+(b1_z-b2_z)**2)
            over = O.bodies[contact.id1].shape.radius + O.bodies[contact.id2].shape.radius - dist
            diam = 1/(1/(O.bodies[contact.id1].shape.radius*2)+1/(O.bodies[contact.id2].shape.radius*2))
            L_over_diam.append(over/diam)
    m_over_diam = np.mean(L_over_diam)
    print('Mean Overlap/Diameter', m_over_diam)

    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n')
    simulation_report.write('Mean Overlap/Diameter ' + str(m_over_diam) + '\n')
    simulation_report.write('IC generation ends\n\n')
    simulation_report.write('Porosity is '+ str(round(porosity(),3))+'(21 % in Tengattini, 2023)\n\n')
    simulation_report.close()
    print("\nPressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds")
    print('next step is the main loading (triaxial)\n')

    # change the vertical pressure applied
    plate_z.state.vel = (0, 0, 0)
    O.engines = O.engines[:-1] + [PyRunner(command='controlWalls()', iterPeriod = 1)]

    # reset plot (IC done, simulation starts)
    plot.reset()
    # save new reference position for walls
    plate_x.state.refPos = plate_x.state.pos
    plate_y.state.refPos = plate_y.state.pos
    plate_z.state.refPos = plate_z.state.pos

    # save data
    saveData()
    
    # compute vertical load 
    vert_pos_load = plate_z.state.refPos*(1-vert_strain_load) 
    # apply vertical load
    i_load = 1
    plate_z.state.pos = plate_z.state.refPos + (vert_pos_load-plate_z.state.refPos)*i_load/n_load

    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 500

    # trackers    
    L_unbalanced_ite = []
    L_confinement_x_ite = []
    L_confinement_y_ite = []
    L_count_bond = []
    L_rel_error_x = []
    L_rel_error_y = []

    # user print
    print('Loading step :', i_load, '/', n_load, '-> ev =', vert_strain_load*i_load/n_load)

#-------------------------------------------------------------------------------

def addPlotData_cementation_ic():
    """
    Save data in plot.
    """
    # add forces applied on walls
    sx = O.forces.f(plate_x.id)[0]/(plate_y.state.pos[1]*plate_z.state.pos[2])
    sy = O.forces.f(plate_y.id)[1]/(plate_x.state.pos[0]*plate_z.state.pos[2])
    sz = O.forces.f(plate_z.id)[2]/(plate_x.state.pos[0]*plate_y.state.pos[1])
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=0,\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_cementation*100 + 1/3*sy/P_cementation*100 + 1/3*sz/P_cementation*100,\
                 strain_x=100*(plate_x.state.pos[0]-plate_x.state.refPos[0])/plate_x.state.refPos[0],
                 strain_y=100*(plate_y.state.pos[1]-plate_y.state.refPos[1])/plate_y.state.refPos[1],
                 strain_z=100*(plate_z.state.pos[2]-plate_z.state.refPos[2])/plate_z.state.refPos[2])

#-------------------------------------------------------------------------------

def addPlotData_confinement_ic():
    """
    Save data in plot.
    """
    # add forces applied on walls
    sx = O.forces.f(plate_x.id)[0]/(plate_y.state.pos[1]*plate_z.state.pos[2])
    sy = O.forces.f(plate_y.id)[1]/(plate_x.state.pos[0]*plate_z.state.pos[2])
    sz = O.forces.f(plate_z.id)[2]/(plate_x.state.pos[0]*plate_y.state.pos[1])
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=count_bond(),\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_confinement*100 + 1/3*sy/P_confinement*100 + 1/3*sz/P_confinement*100, \
                 strain_x=100*(plate_x.state.pos[0]-plate_x.state.refPos[0])/plate_x.state.refPos[0],
                 strain_y=100*(plate_y.state.pos[1]-plate_y.state.refPos[1])/plate_y.state.refPos[1],
                 strain_z=100*(plate_z.state.pos[2]-plate_z.state.refPos[2])/plate_z.state.refPos[2])

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
#Load
#-------------------------------------------------------------------------------

def controlWalls():
    '''
    Control the upper wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fx = O.forces.f(plate_x.id)[0]
    if Fx == 0:
        plate_x.state.pos =  (max([b.state.pos[0]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_y.state.pos[1]/2, plate_z.state.pos[2]/2)
    else :
        dF = Fx - P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_x.state.vel = (np.sign(dF)*v_try_abs, 0, 0)
        else :
            plate_x.state.vel = (np.sign(dF)*v_plate_max, 0, 0)
    
    Fy = O.forces.f(plate_y.id)[1]
    if Fy == 0:
        plate_y.state.pos =  (plate_x.state.pos[0]/2, max([b.state.pos[1]+0.99*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]), plate_z.state.pos[2]/2)
    else :
        dF = Fy - P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2]
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_y.state.vel = (0, np.sign(dF)*v_try_abs, 0)
        else :
            plate_y.state.vel = (0, np.sign(dF)*v_plate_max, 0)

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

def checkUnbalanced():
    """
    Look for the equilibrium during the loading phase.
    """
    global i_load, L_unbalanced_ite, L_confinement_x_ite, L_confinement_y_ite, L_count_bond, L_rel_error_x, L_rel_error_y
    # track and plot unbalanced
    L_unbalanced_ite.append(unbalancedForce())
    # track and plot confinement
    L_confinement_x_ite.append(O.forces.f(plate_x.id)[0]/(P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2])*100)
    L_confinement_y_ite.append(O.forces.f(plate_y.id)[1]/(P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2])*100)
    # track and plot bonds number
    L_count_bond.append(count_bond())

    # plot
    if len(L_unbalanced_ite)>2:
        fig, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize=(16,9),num=1)
        # unbalanced
        ax1.plot(L_unbalanced_ite)
        ax1.set_title('unbalanced force (-)')
        ax1.set_ylim(ymin=0, ymax=2*unbalancedForce_criteria)
        # confinement
        ax2.plot(L_confinement_x_ite)
        ax2.plot(L_confinement_y_ite)
        ax2.set_ylim(ymin=0, ymax=150)
        ax2.set_title('confinements (%)')
        # number of bond
        ax3.plot(L_count_bond)
        ax3.set_title('Number of bond (-)')
        # close
        #fig.savefig('plot/tracking_ite_'+str(i_load)+'.png')
        plt.close()

    # trackers
    L_rel_error_x.append(abs(O.forces.f(plate_x.id)[0] - P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2])/(P_confinement*plate_y.state.pos[1]*plate_z.state.pos[2]))
    L_rel_error_y.append(abs(O.forces.f(plate_y.id)[1] - P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2])/(P_confinement*plate_x.state.pos[0]*plate_z.state.pos[2]))
    if len(L_rel_error_x) < window:
        return
    # check the force applied
    if max(L_rel_error_x[-window:]) > 0.005 or max(L_rel_error_y[-window:]) > 0.005 :
        return
    # verify unbalanced force criteria
    if unbalancedForce() < unbalancedForce_criteria:
        # apply vertical load
        i_load = i_load + 1
        plate_z.state.pos = plate_z.state.refPos + (vert_pos_load-plate_z.state.refPos)*i_load/n_load
 
        # reset trackers
        L_unbalanced_ite = []
        L_confinement_x_ite = []
        L_confinement_y_ite = []
        L_count_bond = []

        # save data
        saveData()
        
        # check simulation stop conditions
        if i_load > n_load:
            stopLoad()
        else :
            # user print
            print('Loading step :', i_load, '/', n_load, '-> ev =', vert_strain_load*i_load/n_load)

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
    save_folder = '../../DEM/Data_Tengattini2023/'+O.tags['d.id']
    os.mkdir(save_folder)
    shutil.copytree('data',save_folder+'/data')
    shutil.copytree('plot',save_folder+'/plot')
    shutil.copy('Tengattini2023.py',save_folder+'/Tengattini2023.py')
    shutil.copy(O.tags['d.id']+'_report.txt',save_folder+'/'+O.tags['d.id']+'_report.txt')

#-------------------------------------------------------------------------------

def addPlotData():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    sx = O.forces.f(plate_x.id)[0]/(plate_y.state.pos[1]*plate_z.state.pos[2])
    sy = O.forces.f(plate_y.id)[1]/(plate_x.state.pos[0]*plate_z.state.pos[2])
    sz = O.forces.f(plate_z.id)[2]/(plate_x.state.pos[0]*plate_y.state.pos[1])
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=count_bond(), \
                Sx=sx, Sy=sy, Sz=sz, \
                X_plate=plate_x.state.pos[0], Y_plate=plate_y.state.pos[1], Z_plate=plate_z.state.pos[2],\
                conf_verified= 1/2*sx/(P_confinement)*100 + 1/2*sy/(P_confinement)*100, \
                strain_x=100*(plate_x.state.pos[0]-plate_x.state.refPos[0])/plate_x.state.refPos[0],\
                strain_y=100*(plate_y.state.pos[1]-plate_y.state.refPos[1])/plate_y.state.refPos[1],\
                strain_z=100*(plate_z.state.pos[2]-plate_z.state.refPos[2])/plate_z.state.refPos[2])

#-------------------------------------------------------------------------------

def saveData():
    """
    Save data in .txt file during the steps.
    """
    addPlotData()
    plot.saveDataTxt('data/'+O.tags['d.id']+'.txt')
    # post-proccess
    L_coordination = []
    L_n_bond = []
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_deviatoric = []
    L_strain_x = []
    L_strain_y = []
    L_strain_z = []
    L_shear_strain = []
    file = 'data/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(data[i][0])
            L_sigma_y.append(data[i][1])
            L_sigma_z.append(data[i][2])
            L_sigma_deviatoric.append(1/2*(L_sigma_z[-1]-L_sigma_x[-1]) + 1/2*(L_sigma_z[-1]-L_sigma_y[-1]))
            L_coordination.append(data[i][7])
            L_n_bond.append(data[i][8])
            L_strain_x.append(abs(data[i][11]))
            L_strain_y.append(abs(data[i][12]))
            L_strain_z.append(abs(data[i][13]))
            L_shear_strain.append(abs(1/2*2/3*(L_strain_z[-1]-L_strain_x[-1]) + 1/2*2/3*(L_strain_z[-1]-L_strain_y[-1])))

        # Add Tengattini 2023 for 500, 1000, 1500 kPa of confinement
        L_strain_z_ref_500         = [0,  0.13,  1.03,   2.10,   2.85,   3.66,   4.82,   6.42,   8.11,   9.76,   11.0]
        L_sigma_deviatoric_ref_500 = [0, 230e3, 678e3, 1166e3, 1423e3, 1449e3, 1399e3, 1336e3, 1248e3, 1250e3, 1200e3]
        L_strain_z_ref_1000         = [0,  0.04,  0.29,   0.74,   1.32,   2.18,   3.32,   4.19,   5.04,   6.80,   8.05,   9.88,  11.48,  13.20]
        L_sigma_deviatoric_ref_1000 = [0, 243e3, 742e3, 1434e3, 2138e3, 2778e3, 3163e3, 3356e3, 3292e3, 3038e3, 2732e3, 2568e3, 2352e3, 2276e3]
        L_strain_z_ref_1500         = [0,   0.85,   1.56,   2.43,   3.25,   4.90,   6.64,   8.20,   8.85,   9.74]
        L_sigma_deviatoric_ref_1500 = [0, 2483e3, 3545e3, 4160e3, 4532e3, 4405e3, 4100e3, 3756e3, 3513e3, 3399e3]

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(16,9),num=1)

        ax1.plot(L_coordination)
        ax1.set_title('Coordination (-)')

        ax2.plot(L_n_bond)
        ax2.set_title('Number of bond (-)')

        # add Tengattini 2023
        ax3.plot(L_strain_z, L_sigma_deviatoric)
        if P_confinement == 0.5e6:
            ax3.plot(L_strain_z_ref_500, L_sigma_deviatoric_ref_500, linestyle='dashed', color='k')
        if P_confinement == 1.0e6:
            ax3.plot(L_strain_z_ref_1000, L_sigma_deviatoric_ref_1000, linestyle='dashed', color='k')
        if P_confinement == 1.5e6:
            ax3.plot(L_strain_z_ref_1500, L_sigma_deviatoric_ref_1500, linestyle='dashed', color='k')
        ax3.set_xlabel(r'$\epsilon_z$ (%)')
        ax3.set_ylabel(r'Deviatoric stress (Pa)')
        
        ax4.plot(L_shear_strain, L_sigma_deviatoric)
        ax4.set_xlabel(r'$\epsilon_q$ (%)')
        ax4.set_ylabel(r'Deviatoric stress (Pa)')

        ax5.plot(L_strain_z, L_sigma_z)
        ax5.set_xlabel(r'$\epsilon_z$ (%)')
        ax5.set_ylabel(r'Vertical stress (Pa)')

        ax6.plot(L_strain_z, L_strain_x, label=r'$\epsilon_x$')
        ax6.plot(L_strain_z, L_strain_y, label=r'$\epsilon_y$')
        ax6.legend(fontsize=15)
        ax6.set_xlabel(r'$\epsilon_z$ (%)')
        ax6.set_ylabel(r'Lateral strain (%)')

        plt.suptitle(r'Trackers - loading step (-)')
        plt.savefig('plot/'+O.tags['d.id']+'.png')
        plt.close()

#-------------------------------------------------------------------------------
# start simulation
#-------------------------------------------------------------------------------

O.run()
waitIfBatch()
