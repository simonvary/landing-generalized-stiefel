import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1

methods_ids = ['rsd', 'land_riem', 'land_precon']

method_names = {
    'rsd' : 'Riem. steepest descent',
    'land_riem' : 'Landing (Riem. gradient)',
    'land_precon' : 'Landing ($\Psi_B(X)$ descent)'
}

#from config import methods_ids, colors, names, line_styles
with open('data/1_gevp.pkl', 'rb') as handle:
    results = pickle.load(handle)

colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]


optlog_rsd = results['optlog_rsd']
optlog_land_riem = results['optlog_land_riem']
optlog_land_precon = results['optlog_land_precon']
obj_true = results['obj_true']

# Objective values plot
plt.figure(figsize=(4, 3), dpi= 220)
plt.semilogy(optlog_rsd['iterations']['time'], optlog_rsd['iterations']['fx'] - obj_true, label = 'Riem. steepest descent',
              linewidth=3, color=colors['rsd'], alpha=0.7)
plt.semilogy(optlog_land_riem['iterations']['time'], optlog_land_riem['iterations']['fx'] - obj_true, label='Landing (Riem. gradient)',
              linewidth=3, color=colors['land_riem'], alpha=0.7)
plt.semilogy(optlog_land_precon['iterations']['time'], optlog_land_precon['iterations']['fx'] - obj_true, 
                label='Landing ($\Psi_B(X)$ descent)', linewidth=3, color=colors['land_precon'], alpha=0.7)
plt.legend()

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('1_gevp_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Distances
plt.figure(figsize=(4, 3), dpi= 220)

plt.semilogy(optlog_land_riem['iterations']['time'], optlog_land_riem['iterations']['distance'], label='Landing (Riem. gradient)',
              linewidth=3, color=colors['land_riem'], alpha=0.7)
plt.semilogy(optlog_land_precon['iterations']['time'], optlog_land_precon['iterations']['distance'], 
                label='Landing ($\Psi_B(X)$ descent)', linewidth=3,
                color=colors['land_precon'], alpha=0.7)
plt.legend()

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('1_gevp_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))



