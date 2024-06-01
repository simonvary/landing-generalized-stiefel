import pickle
import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1

methods_ids = ['land_precon', 'plam']#, 'land_riem']

method_names = {
    'rsd' : 'Riem. grad. descent',
    'land_R' : 'Landing with $\Psi^\mathrm{R}_B(X)$',
    'plam' : 'PLAM',
    'land_precon' : 'Landing with $\Psi_B(X)$',
    'land_riem' : 'Landing with $\mathrm{grad}_{\mathrm{St}_B} f(X)$',
}

max_time = 120 # in seconds
maxiter = 10000
cond_number = 1e2
coef = 1

filename = '4_gevp_c'+str(coef)+'_cond'+str(cond_number)

#from config import methods_ids, colors, names, line_styles
with open('data/' + filename +  '.pkl', 'rb') as handle:
    results = pickle.load(handle)

colors = {}
for i in range(len(methods_ids)):
    colors[methods_ids[i]] = colormap.colors[i]


optlogs = results['optlogs']
obj_true = results['obj_true']

omega_vars = [optlogs[i]['omega_var'] for i in range(len(optlogs))]
eta_vars = [optlogs[i]['eta_var'] for i in range(len(optlogs))]

print(omega_vars)
print(eta_vars)

# Objective values plot vs time
plt.figure(figsize=(4, 3), dpi= 220)
for i in range(len(optlogs)):
    for method_id in methods_ids:
        if i == 0:
            label = method_names[method_id]
        else:
            label = None
        plt.semilogy(optlogs[i][method_id]['iterations']['time'], optlogs[i][method_id]['iterations']['fx'] - obj_true, label = label,linewidth=1, color=colors[method_id], alpha=0.7)
plt.legend()
plt.ylim([1e-16, 1e4])
x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Objective value')
plt.grid()
plt.savefig(filename + '_obj.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs time
plt.figure(figsize=(4, 3), dpi= 220)
for i in range(len(optlogs)):
    for method_id in methods_ids:
        if i == 0:
            label = method_names[method_id]
        else:
            label = None
        plt.semilogy(np.array(optlogs[i][method_id]['iterations']['time']), optlogs[i][method_id]['iterations']['distance'], label = label,linewidth=1, color=colors[method_id], alpha=0.7)
plt.legend()
plt.ylim([1e-16, 1e4])
#plt.xlim([1, 10])
x_ = plt.xlabel('Time (sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig(filename + '_dist.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Distances vs time
plt.figure(figsize=(4, 3), dpi= 220)
for i in range(len(optlogs)):
    for method_id in methods_ids:
        if i == 0:
            label = method_names[method_id]
        else:
            label = None
        plt.loglog(1+np.array(optlogs[i][method_id]['iterations']['time']), optlogs[i][method_id]['iterations']['distance'], label = label,linewidth=1, color=colors[method_id], alpha=0.7)
plt.legend()
plt.ylim([1e-16, 1e4])
#plt.xlim([1, 10])
x_ = plt.xlabel('Time (1 + sec.)')
y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
plt.grid()
plt.savefig(filename + '_dist_loglog.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))
