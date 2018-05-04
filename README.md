# PD_models
Python based PD_models to predict alpha-syn spread in Parkinson's Disease

# Usage
Class `agent_based_model` simulates the interactions and mobility patterns of alpha-syn individually
```python
import numpy as np
from PD_models import agent_based_model
ref_model = agent_based_model(v=1 , N_regions=42, dt=0.01, sconn_len=sconn_len, sconn_den=sconn_den, snca=snca.flatten(), gba=gba.flatten(), roi_size=roi_size.flatten(), fconn=np.zeros((42, 42)), fcscale=0)
# set T for the growth of normal alpha-syn
T = 10000
ref_model = agent_based_model(v=1 , N_regions=42, dt=0.01, sconn_len=sconn_len, sconn_den=sconn_den, snca=snca.flatten(), gba=gba.flatten(), roi_size=roi_size.flatten(), fconn=np.zeros((42, 42)), fcscale=0)
# start the process of normal alpha-syn growth
for t in range(T):
    ref_model.record_to_history()
    ref_model.normal_alpha_syn_growth_edge()
    ref_model.normal_alpha_syn_growth_region()
    print(ref_model.nor)
# initiate the spreading process
ref_model.inject_mis()
T_spread = 20000
for t in range(T_spread):
    ref_model.record_to_history()
    ref_model.misfolded_alpha_syn_spread_edge()
    ref_model.misfolded_alpha_syn_spread_region()
print(ref_model.mis)   


Class `SIR_model` simulates the system based on the differential equations...
```python
import numpy as np
from PD_models import SIR_model
ref_model = agent_based_model(v=1 , N_regions=42, dt=0.01, sconn_len=sconn_len, sconn_den=sconn_den, snca=snca.flatten(), gba=gba.flatten(), roi_size=roi_size.flatten(), fconn=np.zeros((42, 42)), fcscale=0)
# set T for the growth of normal alpha-syn
T = 10000
ref_model = SIR_model(v=1 , N_regions=42, dt=0.01, sconn_len=sconn_len, sconn_den=sconn_den, snca=snca.flatten(), gba=gba.flatten(), roi_size=roi_size.flatten(), fconn=np.zeros((42, 42)), fcscale=0)
# start the process of normal alpha-syn growth
for t in range(T):
    ref_model.record_to_history()
    ref_model.normal_alpha_syn_growth_edge()
    ref_model.normal_alpha_syn_growth_region()
print(ref_model.nor)
# initiate the spreading process 
ref_model.inject_mis()
T_spread = 20000
for t in range(T_spread):
    ref_model.record_to_history()
    ref_model.misfolded_alpha_syn_spread_edge()
    ref_model.misfolded_alpha_syn_spread_region()
print(ref_model.mis)