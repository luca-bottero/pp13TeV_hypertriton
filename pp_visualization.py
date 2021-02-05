#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import uproot
import os
from hipe4ml.tree_handler import TreeHandler

#%%
data = TreeHandler(os.path.abspath(os.getcwd()) + '/data/SignalTable_pp13TeV_mtexp.root', "SignalTable").get_data_frame()

#%%
training_variables = ["pt", "cos_pa" , "tpc_ncls_de" , "tpc_ncls_pr" , "tpc_ncls_pi", "tpc_nsig_de", "tpc_nsig_pr", "tpc_nsig_pi", "dca_de_pr", "dca_de_pi", "dca_pr_pi", "dca_de_sv", "dca_pr_sv", "dca_pi_sv", "chi2"]
print(list(data.columns))
data.head(10)
# %%
#sns.pairplot(data.query('gReconstructed > 0').sample(1000), hue = 'gReconstructed', plot_kws={'alpha': 0.1}, corner = True)
#data.query('gReconstructed > 0').hist(bins = 50, figsize = (20,20));

# %%
rec = data.query('gReconstructed > 0').copy()
rec_rej_acc = rec.query('rej_accept > 0').copy()

print(len(rec)/len(data))
print(len(rec_rej_acc)/len(data))

#%%
## TRASVERSE MOMENTUM COMPARISON PLOT
plt.hist(rec['pt'],bins=100, label='Without Rejection')
plt.hist(rec_rej_acc['pt'],bins=100, label='With Rejection')
plt.title('Trasverse momentum $p_T$', fontsize=15)
plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.legend()
plt.savefig("./images/pt_comparison.png",dpi = 300, facecolor = 'white')
plt.show()

# %%
## CT PLOT
plt.hist(rec_rej_acc['ct'],bins=100, label='With Rejection',log=True,color='orange')
plt.title('ct with rejection', fontsize=15)
plt.xlabel('ct [cm]', fontsize=12)
plt.ylabel('Count',fontsize=12)
plt.savefig("./images/ct_distr_rej.png",dpi = 300, facecolor = 'white')
plt.show()

# %%
# efficiency HISTOGRAMS

hist_rec, bin_edges = np.histogram(rec['pt'], bins=100, density=False)
hist_gen, bin_edges = np.histogram(data['gPt'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue")
plt.title('$p_T$ efficiency', fontsize=15)
plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
plt.ylabel('efficiency', fontsize=12)
plt.savefig('./images/pt_efficiency.png',dpi = 300, facecolor = 'white')
plt.show()

hist_rec, bin_edges = np.histogram(rec_rej_acc['pt'], bins=100, density=False)
hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gPt'], bins=bin_edges, density=False)    #!!!!! should I include .query('rej_accept > 0') ???
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="orange")
plt.title('$p_T$ efficiency with rejection', fontsize=15)
plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
plt.ylabel('efficiency', fontsize=12)
plt.savefig('./images/pt_efficiency_rej.png',dpi = 300, facecolor = 'white')
plt.show()

hist_rec, bin_edges = np.histogram(rec['ct'], bins=100, density=False)
hist_gen, bin_edges = np.histogram(data['gCt'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue")
plt.title('ct efficiency', fontsize=15)
plt.xlabel('ct  [cm]', fontsize=12)
plt.ylabel('efficiency', fontsize=12)
plt.savefig('./images/ct_efficiency.png',dpi = 300, facecolor = 'white')
plt.show()

hist_rec, bin_edges = np.histogram(rec_rej_acc['ct'], bins=100, density=False)
hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gCt'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="orange")
plt.title('ct efficiency with rejection', fontsize=15)
plt.xlabel('ct  [cm]', fontsize=12)
plt.ylabel('efficiency', fontsize=12)
plt.savefig('./images/ct_efficiency_rej.png',dpi = 300, facecolor = 'white')
plt.show()

#%%
# EFFICIENCIES WITH CUTS

query = 'abs(tpc_nsig_de) < 3 & abs(tpc_nsig_pr) < 3 & abs(tpc_nsig_pi) < 3 & cos_pa > '

for cos_pa in (0.99,0.98):
    #pt
    hist_rec, bin_edges = np.histogram(rec_rej_acc.query(query + str(cos_pa))['pt'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gPt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="orange",label='With cuts')
    plt.title('$p_T$, TPC $|n\sigma| < 3$  $\cos(p_a)>$'+str(cos_pa), fontsize=15)
    plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.savefig('./images/pt_eff_rej_cospa_' + str(cos_pa) + '.png',dpi = 300, facecolor = 'white')
    plt.show()

    #pt comparison
    hist_rec, bin_edges = np.histogram(rec_rej_acc['pt'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gPt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue",label='Without cuts')
    
    hist_rec2, bin_edges = np.histogram(rec_rej_acc.query(query + str(cos_pa))['pt'], bins=100, density=False)
    hist_gen2, bin_edges = np.histogram(data.query('rej_accept > 0')['gPt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec2/hist_gen2),width=(bin_edges[1] - bin_edges[0]), color="orange",label='With cuts')
    
    hist_rec, bin_edges = np.histogram(rec_rej_acc['pt'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gPt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen - hist_rec2/hist_gen2),width=(bin_edges[1] - bin_edges[0]), color="red",label='Difference')
    
    plt.title('$p_T$, TPC $|n\sigma| < 3$  $\cos(p_a)>$'+str(cos_pa), fontsize=15)
    plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.legend()
    plt.savefig('./images/pt_eff_rej_cospa_' + str(cos_pa) + 'comparison.png',dpi = 300, facecolor = 'white')
    plt.show()

    #ct
    hist_rec, bin_edges = np.histogram(rec_rej_acc.query(query + str(cos_pa))['ct'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gCt'], bins=bin_edges, density=False)
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="orange")
    plt.title('ct, TPC $|n\sigma| < 3$  $\cos(p_a)>$'+str(cos_pa), fontsize=15)
    plt.xlabel('ct  [cm]', fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.savefig('./images/ct_eff_rej_cospa_' + str(cos_pa) + '.png',dpi = 300, facecolor = 'white')
    plt.show()

    #ct comparison
    hist_rec, bin_edges = np.histogram(rec_rej_acc['ct'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gCt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue",label='Without cuts')
    
    hist_rec2, bin_edges = np.histogram(rec_rej_acc.query(query + str(cos_pa))['ct'], bins=100, density=False)
    hist_gen2, bin_edges = np.histogram(data.query('rej_accept > 0')['gCt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec2/hist_gen2),width=(bin_edges[1] - bin_edges[0]), color="orange",label='With cuts')
    
    hist_rec, bin_edges = np.histogram(rec_rej_acc['ct'], bins=100, density=False)
    hist_gen, bin_edges = np.histogram(data.query('rej_accept > 0')['gCt'], bins=bin_edges, density=False)    
    plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen - hist_rec2/hist_gen2),width=(bin_edges[1] - bin_edges[0]), color="red",label='Difference')
    
    plt.title('ct, TPC $|n\sigma| < 3$  $\cos(p_a)>$'+str(cos_pa), fontsize=15)
    plt.xlabel('ct  [cm]', fontsize=12)
    plt.ylabel('efficiency', fontsize=12)
    plt.legend()
    plt.savefig('./images/ct_eff_rej_cospa_' + str(cos_pa) + 'comparison.png',dpi = 300, facecolor = 'white')
    plt.show()








'''
hist_gen, bin_edges = np.histogram(rec['gPt'], bins=100, density=False)
hist_rec, bin_edges = np.histogram(rec['pt'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue")
plt.title('$p_T$ efficiency', fontsize=15)
plt.xlabel('$p_T$  [GeV/c]', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('./images/pt_efficiency.png',dpi = 300, facecolor = 'white')
plt.show()

hist_gen, bin_edges = np.histogram(rec_rej_acc['gPt'], bins=100, density=False)
hist_rec, bin_edges = np.histogram(rec_rej_acc['pt'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue",facecolor = 'white')
plt.show()

hist_gen, bin_edges = np.histogram(rec['gCt'], bins=100, density=False)
hist_rec, bin_edges = np.histogram(rec['ct'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue",log=True,facecolor = 'white')
plt.show()

hist_gen, bin_edges = np.histogram(rec_rej_acc['gCt'], bins=100, density=False)
hist_rec, bin_edges = np.histogram(rec_rej_acc['ct'], bins=bin_edges, density=False)
plt.bar((bin_edges[1:] + bin_edges[:-1]) * .5, (hist_rec/hist_gen),width=(bin_edges[1] - bin_edges[0]), color="blue",log=True,facecolor = 'white')
plt.show()
'''

# %%
## RESOLUTION HISTOGRAMS

plt.hist(rec['pt'].to_numpy() - rec['gPt'].to_numpy(), bins=100, log=True)
plt.title('$p_T$ resolution', fontsize=15)
plt.xlabel('$p_{T,rec} - p_{T,gen}$  [GeV/c]', fontsize=12)
plt.ylabel('Count', fontsize=12)
mean = np.round(np.mean(rec['pt'].to_numpy() - rec['gPt'].to_numpy()),4)
RMS = np.round(np.sqrt(np.mean(np.square(rec['pt'].to_numpy() - rec['gPt'].to_numpy()))),4)
plt.annotate('Mean: ' + str(mean) + "\nRMS: " + str(RMS) ,xy=(2,10000))
plt.savefig('./images/pt_resolution.png',dpi = 300, facecolor = 'white')
plt.show()

plt.hist(rec_rej_acc['pt'].to_numpy() - rec_rej_acc['gPt'].to_numpy(), bins=100, log=True, color='orange')
plt.title('$p_T$ resolution with rejection', fontsize=15)
plt.xlabel('$p_{T,rec} - p_{T,gen}$  [GeV/c]',fontsize=12)
plt.ylabel('Count',fontsize=12)
mean = np.round(np.mean(rec_rej_acc['pt'].to_numpy() - rec_rej_acc['gPt'].to_numpy()),4)
RMS = np.round(np.sqrt(np.mean(np.square(rec_rej_acc['pt'].to_numpy() - rec_rej_acc['gPt'].to_numpy()))),4)
plt.annotate('Mean: ' + str(mean) + "\nRMS: " + str(RMS) ,xy=(2,10000))
plt.savefig('./images/pt_resolution_rej.png',dpi = 300, facecolor = 'white')
plt.show()

plt.hist(rec['ct'].to_numpy() - rec['gCt'].to_numpy(), bins=100, log=True)
plt.title('ct resolution', fontsize=15)
plt.xlabel('$ct_{rec} - ct_{gen}$  [cm]',fontsize=12)
plt.ylabel('Count',fontsize=12)
mean = np.round(np.mean(rec['pt'].to_numpy() - rec['gPt'].to_numpy()),4)
RMS = np.round(np.sqrt(np.mean(np.square(rec['ct'].to_numpy() - rec['gCt'].to_numpy()))),4)
plt.annotate('Mean: ' + str(mean) + "\nRMS: " + str(RMS) ,xy=(15,10000))
plt.savefig('./images/ct_resolution.png',dpi = 300, facecolor = 'white')
plt.show()

plt.hist(rec_rej_acc['ct'].to_numpy() - rec_rej_acc['gCt'].to_numpy(), bins=100, log=True, color='orange')
plt.title('ct resolution with rejection', fontsize=15)
plt.xlabel('$ct_{rec} - ct_{gen}$  [cm]',fontsize=12)
plt.ylabel('Count',fontsize=12)
mean = np.round(np.mean(rec_rej_acc['ct'].to_numpy() - rec_rej_acc['gCt'].to_numpy()),4)
RMS = np.round(np.sqrt(np.mean(np.square(rec_rej_acc['ct'].to_numpy() - rec_rej_acc['gCt'].to_numpy()))),4)
plt.annotate('Mean: ' + str(mean) + "\nRMS: " + str(RMS) ,xy=(15,10000))
plt.savefig('./images/ct_resolution_rej.png',dpi = 300, facecolor = 'white')
plt.show()


# %%
## SCATTER PLOT OF RESOLUTION

plt.scatter(rec['pt'].to_numpy() - rec['gPt'].to_numpy(),rec['pt'].to_numpy(), alpha=0.1)
plt.xlabel('$p_T$ resolution  [GeV/c]', fontsize=12)
plt.ylabel('$p_{T,rec}$  [GeV/c]', fontsize=12)
plt.title('Reconstructed $p_T$ vs $p_T$ resolution', fontsize=15)
plt.savefig('./images/pt_pt_res.png',dpi = 300, facecolor = 'white')
plt.show()

plt.scatter(rec_rej_acc['pt'].to_numpy() - rec_rej_acc['gPt'].to_numpy(),rec_rej_acc['pt'].to_numpy(), alpha=0.1, color = 'orange')
plt.xlabel('$p_T$ resolution  [GeV/c]', fontsize=12)
plt.ylabel('$p_{T,rec}$  [GeV/c]', fontsize=12)
plt.title('Reconstructed $p_T$ vs $p_T$ resolution with rejection', fontsize=15)
plt.savefig('./images/pt_pt_res_rej.png',dpi = 300, facecolor = 'white')
plt.show()

# %%

## SCATTER PLOT OF REC VS GEN
plt.scatter(rec['pt'].to_numpy(), rec['gPt'].to_numpy(), alpha=0.1, label='Without rejection')
plt.xlabel('$p_{T,gen}$   [GeV/c]', fontsize=12)
plt.ylabel('$p_{T,rec}$  [GeV/c]', fontsize=12)
plt.title('Reconstructed $p_T$ vs generated $p_T$', fontsize=15)
plt.savefig('./images/pt_reg_gen.png',dpi = 300, facecolor = 'white')
plt.show()

plt.scatter(rec_rej_acc['pt'].to_numpy(), rec_rej_acc['gPt'].to_numpy(), alpha=0.1, color='orange', label='With rejection')
plt.xlabel('$p_{T,gen}$   [GeV/c]', fontsize=12)
plt.ylabel('$p_{T,rec}$  [GeV/c]', fontsize=12)
plt.title('Reconstructed $p_T$ vs generated $p_T$ with rejection', fontsize=15)
plt.savefig('./images/pt_reg_gen_rej.png',dpi = 300, facecolor = 'white')
plt.show()

plt.scatter(rec['ct'].to_numpy(), rec['gCt'].to_numpy(), alpha=0.1)
plt.xlabel('$ct_{gen}$   [GeV/c]', fontsize=12)
plt.ylabel('$ct_{rec}$  [GeV/c]', fontsize=12)
plt.title('$ct_{rec}$ vs $ct_{gen}$ without rejection', fontsize=15)
plt.savefig('./images/ct_reg_gen.png',dpi = 300, facecolor = 'white')
plt.show()

plt.scatter(rec_rej_acc['ct'].to_numpy(), rec_rej_acc['gCt'].to_numpy(), alpha=0.1, color='orange')
plt.xlabel('$ct_{gen}$   [GeV/c]', fontsize=12)
plt.ylabel('$ct_{rec}$  [GeV/c]', fontsize=12)
plt.title('$ct_{rec}$ vs $ct_{gen}$ with rejection', fontsize=15)
plt.savefig('./images/ct_reg_gen_rej.png',dpi = 300, facecolor = 'white')
plt.show()

# %%
