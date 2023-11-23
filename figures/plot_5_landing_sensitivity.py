import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'text.usetex' : True})

colormap = plt.cm.Set1

method_id = 'land_precon'
method_name =  'Landing with $\Psi_B(X)$'

#from config import methods_ids, colors, names, line_styles
with open('data/5_gevp_landing_sensitivity.pkl', 'rb') as handle:
    results = pickle.load(handle)

colors = {}

j=0
for i in range(9):
    if i != 5:
        colors[j] = colormap.colors[i]
        j = j+1

optlogs = results['optlogs']
obj_true = results['obj_true']

omega_vars = set([optlogs[i]['omega_var'] for i in range(len(optlogs))])
eta_vars = set([optlogs[i]['eta_var'] for i in range(len(optlogs))])

print(omega_vars)
print(eta_vars)
print(len(optlogs))

for eta_var in eta_vars:
    # Objective values plot vs time
    print(eta_var)
    plt.figure(figsize=(3, 2), dpi= 220)
    j = 0
    for i in range(len(optlogs)):
        if optlogs[i]['eta_var'] == eta_var and optlogs[i]['omega_var']!= 8:
            print(optlogs[i]['omega_var'])
            label = '$' + str(optlogs[i]['omega_var']) + '\omega$'
            plt.semilogy(optlogs[i][method_id]['iterations']['time'], optlogs[i][method_id]['iterations']['fx'] - obj_true, label = label,linewidth=1, color=colors[j], alpha=0.7)
            j = j + 1
    print(j)
    plt.legend(ncol=3, loc='lower left', columnspacing=.5, handlelength=1)
    plt.ylim([1e-12, 1e2])
    x_ = plt.xlabel('Time (sec.)')
    y_ = plt.ylabel('Objective value')
    plt.grid()
    plt.savefig('5_landing_sensitivity_obj_eta' + str(eta_var) +  '.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))

# Distances vs time
for eta_var in eta_vars:
    # Objective values plot vs time
    print(eta_var)
    plt.figure(figsize=(3, 2), dpi= 220)
    j = 0
    for i in range(len(optlogs)):
        if optlogs[i]['eta_var'] == eta_var and optlogs[i]['omega_var']!= 8:
            print(optlogs[i]['omega_var'])
            label = '$' + str(optlogs[i]['omega_var']) + '\omega$'
            plt.semilogy(optlogs[i][method_id]['iterations']['time'], optlogs[i][method_id]['iterations']['distance'], label = label,linewidth=2, color=colors[j], alpha=1)
            j = j + 1
    print(j)
    plt.legend(ncol=3, loc='lower left', columnspacing=.5, handlelength=2)
    plt.ylim([1e-12, 1e2])
    x_ = plt.xlabel('Time (sec.)')
    y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
    plt.grid()
    plt.savefig('5_landing_sensitivity_dist_eta' + str(eta_var) +  '.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))


# Distances vs time
for eta_var in eta_vars:
    # Objective values plot vs time
    print(eta_var)
    plt.figure(figsize=(3, 2), dpi= 220)
    j = 0
    for i in range(len(optlogs)):
        if optlogs[i]['eta_var'] == eta_var and optlogs[i]['omega_var']!= 8:
            print(optlogs[i]['omega_var'])
            label = '$' + str(optlogs[i]['omega_var']) + '\omega$'
            plt.semilogy(optlogs[i][method_id]['iterations']['time'], optlogs[i][method_id]['iterations']['distance'], label = label,linewidth=2, color=colors[j], alpha=1)
            j = j + 1
    print(j)
    plt.legend(ncol=3, loc='lower left', columnspacing=.5, handlelength=1)
    plt.ylim([1e-12, 1e2])
    x_ = plt.xlabel('Time (sec.)')
    y_ = plt.ylabel('Distance $\mathcal{N}(x)$')
    plt.grid()
    plt.savefig('5_landing_sensitivity_dist_loglog_eta' + str(eta_var) +  '.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))
