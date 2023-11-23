plt.rcParams.update({'text.usetex' : True})
colormap = plt.cm.Set1

plt.figure(figsize=(4, 3), dpi= 220)
for i, label in enumerate(["$\Psi_B(X)$", "$\Psi^\mathrm{R}_B(X)$"]):
    times = [np.mean(t[:, i]) for t in time_dirs]
    plt.scatter(n_list, times, label=label)

plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.xticks(n_list, n_list, rotation=30)
#plt.ylim([0.5*1e-4, 1e-4])
x_ = plt.xlabel('Dimension ($n$)')
y_ = plt.ylabel('Time (sec.)')

plt.grid()
plt.savefig('3_cost_relative_grad_cuda.pdf', bbox_inches='tight', bbox_extra_artists=(x_, y_))