import pickle
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1
figsize = figsize = (4, 2)

methods_ids = ['rsd', 'land_R', 'land_precon', 'plam']#, 'land_riem']

method_names = {
    'rsd' : 'Riem. grad. descent',
    'land_R' : 'Landing with $\Psi^\mathrm{R}_B(X)$',
    'plam' : 'PLAM',
    'land_precon' : 'Landing with $\Psi_B(X)$',
    'land_riem' : 'Landing with $\mathrm{grad}_{\mathrm{St}_B} f(X)$',
}

#from config import methods_ids, colors, names, line_styles
with open('data/1_gevp.pkl', 'rb') as handle:
    results = pickle.load(handle)

colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]

optlog={}
optlog['rsd'] = results['optlog_rsd']
optlog['plam'] = results['optlog_plam']
optlog['land_precon'] = results['optlog_land_precon']
optlog['land_R'] = results['optlog_land_R']
optlog['land_riem'] = results['optlog_land_riem']
obj_true = results['obj_true']

# Objective values plot vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    plt.semilogy(optlog[method_id]['iterations']['time'], optlog[method_id]['iterations']['fx'] - obj_true, label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)
plt.legend()

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('1_gevp_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Objective values plot vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    plt.semilogy(optlog[method_id]['iterations']['fx'] - obj_true, label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)
plt.legend()

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('1_gevp_obj_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))



# Distances vs time
plt.figure(figsize=figsize, dpi= 220)

for method_id in methods_ids[1:]:
    plt.semilogy(optlog[method_id]['iterations']['time'], optlog[method_id]['iterations']['distance'], label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)
#plt.legend()

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('1_gevp_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs iterations
plt.figure(figsize=figsize, dpi= 220)

for method_id in methods_ids[1:]:
    plt.semilogy(optlog[method_id]['iterations']['distance'], label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)
#plt.legend()

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('1_gevp_dist_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))



# Safe step region
plt.figure(figsize=figsize, dpi= 220)

for method_id in methods_ids[1:3]:
    safe_steps = [ele.get() for ele in optlog[method_id]['iterations']['safe_step'] ]
    plt.semilogy(optlog[method_id]['iterations']['time'], safe_steps, label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)
plt.legend()

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Safe step-size $\eta(x)$')
plt.grid()
plt.savefig('1_gevp_safestep.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Combined plot

fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True), figsize=(4,3), dpi= 220)

# Objective values plot vs time
for method_id in methods_ids:
    ax1.semilogy(optlog[method_id]['iterations']['time'], optlog[method_id]['iterations']['fx'] - obj_true, label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)

y_ = ax1.set_ylabel('Objective value')
ax1.grid()

for method_id in methods_ids[1:]:
    ax2.semilogy(optlog[method_id]['iterations']['time'], optlog[method_id]['iterations']['distance'], label = method_names[method_id],linewidth=3, color=colors[method_id], alpha=0.7)

x_ = ax2.set_xlabel('Time (sec.)')
y_ = ax2.set_ylabel('Distance $\mathcal{N}(x)$')
ax2.grid()

plt.subplots_adjust(hspace=.1)
ax1.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=1)

plt.savefig('1_gevp_combined.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))
