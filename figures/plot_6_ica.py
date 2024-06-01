import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker

plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1
alpha = .8
methods_ids = ['rrsd', 'land_plam_avg', 'land_precon_avg', 'land_precon']

ylim_dist = [2*1e-7, 1]
ylim_amari  = [.8*1e-4, 3]
ylim_obj = [3.25, 3.75]
figsize = (4, 1.5)
xlim_time = [-0.05, 2.35]
linewidth = 3

method_names = {
    'rrsd' : 'Riem. GD (avg.)',
    'land_plam_avg' : 'PLAM (avg.)',
    'land_precon_avg' : 'Landing ($\Psi_B(X)$, avg.)',
    'land_precon' : 'Landing ($\Psi_B(X)$, online)'
}
coef = 4
n_features = 10

filename = '6_ica_c'+str(coef)+'_n'+str(n_features)
filename = '6_ica_cS_n'+str(n_features)

print(filename+'.pkl')
#from config import methods_ids, colors, names, line_styles
with open('data/'+filename+'.pkl', 'rb') as handle:
    results = pickle.load(handle)
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]




# Objective values plot vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(out['time'],(np.array(out['fx'])), label = method_names[method_id],
              linewidth=linewidth, color=colors[method_id], alpha=alpha)
ax=plt.gca()
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.set_ylim(ylim_obj)
x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)
plt.xlim(xlim_time)
plt.savefig(filename+'_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Objective values plot vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy((np.array(out['fx'])), label = method_names[method_id],
              linewidth=linewidth, color=colors[method_id], alpha=alpha)
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)
ax=plt.gca()
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Objective value')
plt.grid()
ax.set_ylim(ylim_obj)
plt.savefig(filename+'_obj_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Distances vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(out['time'], np.array(out['distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.ylim(ylim_dist)
plt.xlim(xlim_time)
plt.grid()
plt.savefig(filename+'_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(np.array(out['distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.ylim(ylim_dist)
plt.grid()
plt.savefig(filename+'_dist_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Amari distances vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(out['time'], np.array(out['amari_distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)
x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Amari distance')
plt.ylim(ylim_amari)
plt.xlim(xlim_time)
plt.grid()
plt.savefig(filename+'_amari_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Amari distances vs iterations
plt.figure(figsize=figsize, dpi=220)
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(np.array(out['amari_distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)
x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Amari distance')
plt.ylim(ylim_amari)
plt.grid()
plt.savefig(filename+'_amari_dist_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Combined plot
fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True, subplot_kw=dict(frameon=True), figsize=(4,5), dpi= 220)

# obj
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    ax1.semilogy(out['time'],(np.array(out['fx'])), label = method_names[method_id],
              linewidth=linewidth, color=colors[method_id], alpha=alpha)
ax1.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax1.set_ylim(ylim_obj)
ax1.set_xlim(xlim_time)
y_ = ax1.set_ylabel('Objective value')
ax1.grid()

# amari
for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    ax2.semilogy(out['time'], np.array(out['amari_distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)
y_ = ax2.set_ylabel('Amari distance')
ax2.grid()
ax2.set_ylim(ylim_amari)
ax2.set_xlim(xlim_time)
#plt.tick_params('x', labelsize = 6)

for method_id in methods_ids:
    out = results[method_id]
    #obj_true = results['obj_true']
    plt.semilogy(out['time'], np.array(out['distance']), label = method_names[method_id], linewidth=linewidth, color=colors[method_id], alpha=alpha)
ax3.grid()
x_ = ax3.set_xlabel('Time (sec.)')
y_ = ax3.set_ylabel('Distance $\mathcal{N}(x)$')
ax3.set_ylim(ylim_dist)
ax3.set_xlim(xlim_time)

ax2.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=1)

plt.subplots_adjust(hspace=.1)
plt.savefig(filename+'_combined.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))
