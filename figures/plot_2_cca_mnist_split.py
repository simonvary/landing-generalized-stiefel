import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1
alpha = 1
methods_ids = ['rrsd', 'land_precon_avg', 'land_precon']
figsize = (4,2)

method_names = {
    'rrsd' : 'Riem. GD (avg.)',
    'land_precon_avg' : 'Landing ($\Psi_B(X)$, avg.)',
    'land_precon' : 'Landing ($\Psi_B(X)$, online)'
}

#from config import methods_ids, colors, names, line_styles
with open('data/2_cca_mnist_split_p5.pkl', 'rb') as handle:
    results = pickle.load(handle)
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]


# Objective values plot vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    plt.semilogy(out['time'],-(np.array(out['fx']) - obj_true)/obj_true, label = method_names[method_id],
              linewidth=3, color=colors[method_id], alpha=alpha)
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('2_cca_mnist_split_p5_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Objective values plot vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    plt.semilogy(-(np.array(out['fx']) - obj_true)/obj_true, label = method_names[method_id],
              linewidth=3, color=colors[method_id], alpha=alpha)
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('2_cca_mnist_split_p5_obj_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Distances vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    plt.semilogy(out['time'], np.array(out['distanceA']) + np.array(out['distanceB']), label = method_names[method_id], linewidth=3, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('2_cca_mnist_split_p5_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    plt.semilogy(np.array(out['distanceA']) + np.array(out['distanceB']), label = method_names[method_id], linewidth=3, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('2_cca_mnist_split_p5_dist_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))





###### p = 10

#from config import methods_ids, colors, names, line_styles
with open('data/2_cca_mnist_split_p10.pkl', 'rb') as handle:
    results = pickle.load(handle)
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]
# Objective values plot vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    # out['time'],
    plt.semilogy(out['time'], -(np.array(out['fx']) - obj_true)/obj_true, label = method_names[method_id],
              linewidth=3, color=colors[method_id], alpha=alpha)
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('2_cca_mnist_split_p10_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Objective values plot vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    # out['time'],
    plt.semilogy(-(np.array(out['fx']) - obj_true)/obj_true, label = method_names[method_id],
              linewidth=3, color=colors[method_id], alpha=alpha)
plt.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=2)

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig('2_cca_mnist_split_p10_obj_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs time
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    # out['time']
    plt.semilogy(out['time'], np.array(out['distanceA']) + np.array(out['distanceB']), label = method_names[method_id], linewidth=3, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('2_cca_mnist_split_p10_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))



# Distances vs iterations
plt.figure(figsize=figsize, dpi= 220)
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    # out['time']
    plt.semilogy(np.array(out['distanceA']) + np.array(out['distanceB']), label = method_names[method_id], linewidth=3, color=colors[method_id], alpha=alpha)

x_ = plt.xlabel('Iterations')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig('2_cca_mnist_split_p10_dist_iter.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))




# Combined plot for p = 5

#from config import methods_ids, colors, names, line_styles
with open('data/2_cca_mnist_split_p5.pkl', 'rb') as handle:
    results = pickle.load(handle)
colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]



fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True), figsize=(4,3), dpi= 220)

for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    ax1.semilogy(out['time'],-(np.array(out['fx']) - obj_true)/obj_true, label = method_names[method_id],
              linewidth=3, color=colors[method_id], alpha=alpha)
y_ = ax1.set_ylabel('Objective value')
ax1.grid()

# Distances vs time
for method_id in methods_ids:
    out = results[method_id]
    obj_true = results['obj_true']
    ax2.semilogy(out['time'], np.array(out['distanceA']) + np.array(out['distanceB']), label = method_names[method_id], linewidth=3, color=colors[method_id], alpha=alpha)

x_ = ax2.set_xlabel('Time (sec.)')
y_ = ax2.set_ylabel('Distance $\mathcal{N}(x)$')
ax2.grid()

plt.subplots_adjust(hspace=.1)
ax1.legend(ncol=1, loc='upper right', columnspacing=.5, handlelength=1)

plt.savefig('2_cca_mnist_split_p5_combined.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))
